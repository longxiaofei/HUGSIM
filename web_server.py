import sys
import os
import pickle
import json
import threading
import time
import io
import enum
from collections import deque, OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict
sys.path.append(os.getcwd())

from fastapi import FastAPI, Body, Header, HTTPException
from fastapi.responses import HTMLResponse, Response
from omegaconf import OmegaConf
from huggingface_hub import HfApi, hf_hub_download
import open3d as o3d
import numpy as np
import gymnasium
import uvicorn

from sim.utils.sim_utils import traj2control, traj_transform_to_global
from sim.utils.score_calculator import hugsim_evaluate

IN_HUGGINGFACE_SPACE = os.getenv('IN_HUGGINGFACE_SPACE', 'false') == 'true'
STOP_SPACE_TIMEOUT = int(os.getenv('STOP_SPACE_TIMEOUT', '7200'))
HF_TOKEN = os.getenv('HF_TOKEN', None)
SPACE_PARAMS = json.loads(os.getenv('PARAMS', '{}'))


class GlobalState:
    done = False


class SubmissionStatus(enum.Enum):
    PENDING = 0
    QUEUED = 1
    PROCESSING = 2
    SUCCESS = 3
    FAILED = 4


def download_submission_info() -> Dict[str. Any]:
    """
    Download the submission info from Hugging Face Hub.
    Args:
        team_id (str): The team ID.
    Returns:
        Dict[str, Any]: The submission info.
    """
    submission_info_path = hf_hub_download(
        repo_id=SPACE_PARAMS["competition_id"],
        filename=f"submission_info/{SPACE_PARAMS["team_id"]}.json",
        repo_type="dataset",
        token=HF_TOKEN
    )
    with open(submission_info_path, 'r') as f:
        submission_info = json.load(f)
    
    return submission_info


def upload_submission_info(user_submission_info: Dict[str, Any]):
    user_submission_info_json = json.dumps(user_submission_info, indent=4)
    user_submission_info_json_bytes = user_submission_info_json.encode("utf-8")
    user_submission_info_json_buffer = io.BytesIO(user_submission_info_json_bytes)
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=user_submission_info_json_buffer,
        path_in_repo=f"submission_info/{SPACE_PARAMS["team_id"]}.json",
        repo_id=SPACE_PARAMS["competition_id"],
        repo_type="dataset",
    )


def update_submission_status(status):
    user_submission_info = download_submission_info()
    for submission in user_submission_info["submissions"]:
        if submission["submission_id"] == SPACE_PARAMS["submission_id"]:
            submission["status"] = status
            break
    upload_submission_info(user_submission_info)


def auto_stop():
    """
    Automatically stop the server after a certain timeout.
    """
    stop_deadline = datetime.now() + timedelta(seconds=STOP_SPACE_TIMEOUT)
    while 1:
        if datetime.now() > stop_deadline:
            update_submission_status(SubmissionStatus.FAILED.value)
            break
        if GlobalState.done:
            update_submission_status(SubmissionStatus.SUCCESS.value)
            break
        time.sleep(60)

    server_space_id = SPACE_PARAMS["server_space_id"]
    client_space_id = SPACE_PARAMS["client_space_id"]
    api = HfApi(token=HF_TOKEN)
    api.delete_repo(
        repo_id=server_space_id,
        repo_type="space"
    )
    api.delete_repo(
        repo_id=client_space_id,
        repo_type="space"
    )

if IN_HUGGINGFACE_SPACE:
    # Start a thread to automatically stop the server after a timeout
    auto_stop_thread = threading.Thread(target=auto_stop, daemon=True)
    auto_stop_thread.start()
    update_submission_status(SubmissionStatus.PROCESSING.value)


class FifoDict:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._order_dict = OrderedDict()
        self.locker = threading.Lock()
    
    def push(self, key: str, value: Any):
        with self.locker:
            if key in self._order_dict:
                self._order_dict.move_to_end(key)
                return
            if len(self._order_dict) >= self.max_size:
                self._order_dict.popitem(last=False)
            self._order_dict[key] = value
    
    def get(self, key: str) -> Any:
        return self._order_dict.get(key, None)


class EnvHandler:
    def __init__(self, cfg, output):
        self.cfg = cfg
        self.output = output
        self.env = gymnasium.make('hugsim_env/HUGSim-v0', cfg=cfg, output=output)
        self.reset_env()
        self._lock = threading.Lock()

    def reset_env(self):
        """
        Reset the environment and initialize variables.
        """
        self._cnt = 0
        self._done = False
        self._save_data = {'type': 'closeloop', 'frames': []}
        self._obs, self._info = self.env.reset()
        self._log_list = deque(maxlen=100)
        self._log("Environment reset complete.")
    
    def get_current_state(self):
        """
        Get the current state of the environment.
        """
        return {
            "obs": self._obs,
            "info": self._info,
        }

    @property
    def has_done(self) -> bool:
        """
        Check if the episode is done.
        Returns:
            bool: True if the episode is done, False otherwise.
        """
        return self._done

    @property
    def log_list(self) -> deque:
        """
        Get the log list.
        Returns:
            deque: The log list containing recent log messages.
        """
        return self._log_list

    def execute_action(self, plan_traj: np.ndarray) -> bool:
        """
        Execute the action based on the planned trajectory.
        Args:
            plan_traj (Any): The planned trajectory to follow.
        Returns:
            bool: True if the episode is done, False otherwise.
        """
        acc, steer_rate = traj2control(plan_traj, self._info)
        action = {'acc': acc, 'steer_rate': steer_rate}
        self._log("Executing action:", action)
    
        self._obs, _, terminated, truncated, self._info = self.env.step(action)
        self._cnt += 1
        self._done = terminated or truncated or self._cnt > 400

        imu_plan_traj = plan_traj[:, [1, 0]]
        imu_plan_traj[:, 1] *= -1
        global_traj = traj_transform_to_global(imu_plan_traj, self._info['ego_box'])
        self._save_data['frames'].append({
            'time_stamp': self._info['timestamp'],
            'is_key_frame': True,
            'ego_box': self._info['ego_box'],
            'obj_boxes': self._info['obj_boxes'],
            'obj_names': ['car' for _ in self._info['obj_boxes']],
            'planned_traj': {
                'traj': global_traj,
                'timestep': 0.5
            },
            'collision': self._info['collision'],
            'rc': self._info['rc']
        })
        
        if not self._done:
            return False

        with open(os.path.join(self.output, 'data.pkl'), 'wb') as wf:
            pickle.dump([self._save_data], wf)
        
        ground_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'ground.ply')).points)
        scene_xyz = np.asarray(o3d.io.read_point_cloud(os.path.join(output, 'scene.ply')).points)
        results = hugsim_evaluate([self._save_data], ground_xyz, scene_xyz)
        with open(os.path.join(output, 'eval.json'), 'w') as f:
            json.dump(results, f)
        
        self._log("Evaluation results saved.")
        return True

    def _log(self, *messages):
        log_message = f"[{str(datetime.now())}]" + " ".join([str(msg) for msg in messages]) + "\n"
        with self._lock:
            self._log_list.append(log_message)


class WebServer:
    def __init__(self, env_handler: EnvHandler, auth_token: str):
        self.env_handler = env_handler
        self.auth_token = auth_token
        self._init_app()
        self._result_dict= FifoDict(max_size=30)
    
    def run(self):
        uvicorn.run(self._app, host="0.0.0.0", port=7860, workers=1)

    def _reset_endpoint(self):
        self.env_handler.reset_env()
        return {"success": True}

    def _get_current_state_endpoint(self):
        state = self.env_handler.get_current_state()
        return Response(content=pickle.dumps(state), media_type="application/octet-stream")

    def _load_numpy_ndarray_json_str(self, json_str: str) -> np.ndarray:
        """
        Load a numpy ndarray from a JSON string.
        """
        data = json.loads(json_str)
        return np.array(data["data"], dtype=data["dtype"]).reshape(data["shape"])

    def _execute_action_endpoint(
        self,
        plan_traj: str = Body(..., embed=True),
        transaction_id: str = Body(..., embed=True),
    ):
        cache_result = self._result_dict.get(transaction_id)
        if cache_result is not None:
            return Response(content=cache_result, media_type="application/octet-stream")

        if self.env_handler.has_done:
            result = pickle.dumps({"done": done, "state": None})
            self._result_dict.push(transaction_id, result)
            return Response(content=result, media_type="application/octet-stream")

        plan_traj = self._load_numpy_ndarray_json_str(plan_traj)
        done = self.env_handler.execute_action(plan_traj)
        GlobalState.done = done
        if done:
            result = pickle.dumps({"done": done, "state": None})
            self._result_dict.push(transaction_id, result)
            return Response(content=result, media_type="application/octet-stream")
        
        state = self.env_handler.get_current_state()
        result = pickle.dumps({"done": done, "state": state})
        self._result_dict.push(transaction_id, result)
        return Response(content=result, media_type="application/octet-stream")

    def _main_page_endpoint(self):
        html_content = f"""
            <html><body><pre>{"\n".join(self.env_handler.log_list)}</pre></body></html>
            <script>
                setTimeout(function() {{
                    window.location.reload();
                }}, 5000);
            </script>
        """
        return HTMLResponse(content=html_content)

    def _verify_token(self, auth_token: str = Header(...)):
        if self.auth_token and self.auth_token != auth_token:
            raise HTTPException(status_code=401)

    def _init_app(self):
        self._app = FastAPI()
        self._app.add_api_route("/reset", self._reset_endpoint, methods=["POST"], dependencies=[self._verify_token])
        self._app.add_api_route("/get_current_state", self._get_current_state_endpoint, methods=["GET"], dependencies=[self._verify_token])
        self._app.add_api_route("/execute_action", self._execute_action_endpoint, methods=["POST"], dependencies=[self._verify_token])
        self._app.add_api_route("/", self._main_page_endpoint, methods=["GET"])


# TODO: add code to update submission info
if __name__ == "__main__":
    # Using fixed paths for web server
    ad = "uniad"
    base_path = os.path.join(os.path.dirname(__file__), 'docker', "web_server_config", 'nuscenes_base.yaml')
    # unknown config
    scenario_path = os.path.join(os.path.dirname(__file__), 'docker', "web_server_config", 'nuscenes_scenario.yaml')
    camera_path = os.path.join(os.path.dirname(__file__), 'docker', "web_server_config", 'nuscenes_camera.yaml')
    kinematic_path = os.path.join(os.path.dirname(__file__), 'docker', "web_server_config", 'kinematic.yaml')

    scenario_config = OmegaConf.load(scenario_path)
    base_config = OmegaConf.load(base_path)
    camera_config = OmegaConf.load(camera_path)
    kinematic_config = OmegaConf.load(kinematic_path)
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config}
    )
    cfg.base.output_dir = cfg.base.output_dir + ad

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    cfg.update(model_config)
    
    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)
    
    env_handler = EnvHandler(cfg, output)
    web_server = WebServer(env_handler, auth_token=os.getenv('HUGSIM_AUTH_TOKEN'))
    web_server.run()
