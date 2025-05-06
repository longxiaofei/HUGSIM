import sys
import os
import pickle
import json
import threading
from collections import deque, OrderedDict
from datetime import datetime
from typing import Any
sys.path.append(os.getcwd())

from fastapi import FastAPI, Body, Header, HTTPException
from fastapi.responses import HTMLResponse
from omegaconf import OmegaConf
import open3d as o3d
import numpy as np
import gymnasium
import uvicorn

from sim.utils.sim_utils import traj2control, traj_transform_to_global
from sim.utils.score_calculator import hugsim_evaluate


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

    def execute_action(self, plan_traj: Any) -> bool:
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
        return {"success": True, "data": pickle.dumps(state["obs"], state["info"])}

    def _execute_action_endpoint(
        self,
        plan_traj: str = Body(..., embed=True),
        transaction_id: str = Body(..., embed=True),
    ):
        cache_result = self._result_dict.get(transaction_id)
        if cache_result is not None:
            return {
                "success": True,
                "data": cache_result,
            }

        if self.env_handler.has_done:
            result = {"done": done, "state": None}
            self._result_dict.push(transaction_id, result)
            return {"success": True, "data": result}

        plan_traj = pickle.loads(plan_traj)
        done = self.env_handler.execute_action(plan_traj)
        if done:
            result = {"done": done, "state": None}
            self._result_dict.push(transaction_id, result)
            return {"success": True, "data": result}
        
        state = self.env_handler.get_current_state()
        result = {"done": done, "state": pickle.dumps(state["obs"], state["info"])}
        self._result_dict.push(transaction_id, result)
        return {
            "success": True,
            "data": {"done": done, "state": result}
        }

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
