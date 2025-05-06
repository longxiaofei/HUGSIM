from typing import Optional, Dict, Any
import pickle
import os
import uuid

from requests import Session
import tenacity


class HugsimClient:
    def __init__(self, host: Optional[str]=None, api_token: Optional[str]=None):
        self.host = host or os.getenv('HUGSIM_SERVER_HOST', 'http://localhost:8065')
        self.api_token = api_token or os.getenv('HUGSIM_API_TOKEN', None)
        self._session = Session()
        self._header = {"auth-token": self.api_token}
    
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
        resp_json = response.json()
        state_data = pickle.loads(resp_json['data'])
        return {
            "obs": state_data[0],
            "info": state_data[1]
        }

    def execute_action(self, plan_traj: Any) -> Dict[str, Any]:
        """
        Execute an action in the environment
        :param plan_traj: The planned trajectory to execute
        :return: A dictionary containing the done status and the state
        """
        transaction_id = uuid.uuid4().hex
        return self._execute_action(plan_traj, transaction_id)

    @tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(5))
    def _execute_action(self, plan_traj: Any, transaction_id: str) -> Dict[str, Any]:
        """
        Execute an action in the environment
        :param plan_traj: The planned trajectory to execute
        :return: A dictionary containing the done status and the state
        """
        url = f"{self.host}/execute_action"
        data = pickle.dumps(plan_traj)
        response = self._session.post(url, headers=self._header, json={"plan_traj": data, "transaction_id": transaction_id})
        if response.status_code != 200:
            raise Exception(f"Failed to execute action: {response.text}")
        resp_json = response.json()
        data = resp_json['data']
        if data['done']:
            return {"done": True, "state": None}
        state_data = pickle.loads(data['state'])
        return {
            "done": False,
            "state": {
                "obs": state_data[0],
                "info": state_data[1]
            }
        }
