import sys
import os
import pickle
import json
import threading
from collections import deque
from datetime import datetime
from typing import Any
sys.path.append(os.getcwd())

from fastapi import FastAPI, Body, Header, HTTPException
from fastapi.responses import HTMLResponse
from argparse import ArgumentParser
from omegaconf import OmegaConf
import open3d as o3d
import numpy as np
import gymnasium
import uvicorn

from sim.utils.sim_utils import traj2control, traj_transform_to_global
from sim.utils.score_calculator import hugsim_evaluate


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
    
    def run(self):
        uvicorn.run(self._app, host="0.0.0.0", port=7860, workers=1)

    def _reset_endpoint(self):
        self.env_handler.reset_env()
        return {"success": True}

    def _get_current_state_endpoint(self):
        state = self.env_handler.get_current_state()
        return {"success": True, "data": pickle.dumps(state["obs"], state["info"])}

    # TODO: add idepotency
    def _execute_action_endpoint(self, plan_traj: str = Body(..., embed=True)):
        if self.env_handler.has_done:
            return {"success": True, "data": {"done": done, "state": None}}

        plan_traj = pickle.loads(plan_traj)
        done = self.env_handler.execute_action(plan_traj)
        if done:
            return {"success": True, "data": {"done": done, "state": None}}
        
        state = self.env_handler.get_current_state()
        return {
            "success": True,
            "data": {"done": done, "state": pickle.dumps(state["obs"], state["info"])}
        }

    def _main_page(self):
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
        self._app.add_api_route("/reset", self.reset_env, methods=["POST"], dependencies=[self._verify_token])
        self._app.add_api_route("/get_current_state", self.get_state, methods=["GET"], dependencies=[self._verify_token])
        self._app.add_api_route("/execute_action", self.execute_action, methods=["POST"], dependencies=[self._verify_token])
        self._app.add_api_route("/", self._main_page, methods=["GET"])


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--scenario_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--camera_path", type=str, required=True)
    parser.add_argument("--kinematic_path", type=str, required=True)
    parser.add_argument('--ad', default="uniad")
    parser.add_argument('--ad_cuda', default="1")
    args = parser.parse_args()

    scenario_config = OmegaConf.load(args.scenario_path)
    base_config = OmegaConf.load(args.base_path)
    camera_config = OmegaConf.load(args.camera_path)
    kinematic_config = OmegaConf.load(args.kinematic_path)
    cfg = OmegaConf.merge(
        {"scenario": scenario_config},
        {"base": base_config},
        {"camera": camera_config},
        {"kinematic": kinematic_config}
    )
    cfg.base.output_dir = cfg.base.output_dir + args.ad

    model_path = os.path.join(cfg.base.model_base, cfg.scenario.scene_name)
    model_config = OmegaConf.load(os.path.join(model_path, 'cfg.yaml'))
    cfg.update(model_config)
    
    output = os.path.join(cfg.base.output_dir, cfg.scenario.scene_name+"_"+cfg.scenario.mode)
    os.makedirs(output, exist_ok=True)

    if args.ad == 'uniad':
        ad_path = cfg.base.uniad_path
    elif args.ad == 'vad':
        ad_path = cfg.base.vad_path
    elif args.ad == 'ltf':
        ad_path = cfg.base.ltf_path
    else:
        raise NotImplementedError
    
    env_handler = EnvHandler(cfg, output)
    web_server = WebServer(env_handler, auth_token=os.getenv('HUGSIM_AUTH_TOKEN'))
    web_server.run()
