from typing import Optional, Dict, Any
import pickle
import os
import uuid
import json

from requests import Session
import numpy as np
import tenacity


class HugsimClient:
    def __init__(self, host: Optional[str]=None, api_token: Optional[str]=None):
        self.host = host or os.getenv('HUGSIM_SERVER_HOST', 'http://localhost:8065')
        self.api_token = api_token or os.getenv('HUGSIM_API_TOKEN', None)
        self._session = Session()
        self._header = {"auth-token": self.api_token}

    def _dump_numpy_ndarray_json_str(self, data: np.ndarray) -> str:
        """
        Convert numpy ndarray to JSON string
        :param data: Numpy ndarray
        :return: JSON string
        """
        info = {
            "data": data.tolist(),
            "shape": data.shape,
            "dtype": str(data.dtype),
        }
        return json.dumps(info)

    def reset_env(self):
        """
        Reset the environment
        """
        url = f"{self.host}/reset"
        response = self._session.post(url, headers=self._header)
        if response.status_code != 200:
            raise Exception(f"Failed to reset environment: {response.text}")

    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current state of the environment
        :return: A dictionary containing the observation and info
        """
        url = f"{self.host}/get_current_state"
        response = self._session.get(url, headers=self._header)
        if response.status_code != 200:
            raise Exception(f"Failed to get current state: {response.text}")
        return pickle.loads(response.content)

    def execute_action(self, plan_traj: np.ndarray) -> Dict[str, Any]:
        """
        Execute an action in the environment
        :param plan_traj: The planned trajectory to execute
        :return: A dictionary containing the done status and the state
        """
        transaction_id = uuid.uuid4().hex
        return self._execute_action(plan_traj, transaction_id)

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(5))
    def _execute_action(self, plan_traj: np.ndarray, transaction_id: str) -> Dict[str, Any]:
        """
        Execute an action in the environment
        :param plan_traj: The planned trajectory to execute
        :return: A dictionary containing the done status and the state
        """
        url = f"{self.host}/execute_action"
        response = self._session.post(
            url,
            headers=self._header,
            json={"plan_traj": self._dump_numpy_ndarray_json_str(plan_traj), "transaction_id": transaction_id})
        if response.status_code != 200:
            raise Exception(f"Failed to execute action: {response.text}")
        data = pickle.loads(response.content)
        return data
