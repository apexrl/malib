import gym

from malib.envs import Environment
from malib.utils.typing import Dict, Any, AgentID
from malib.backend.datapool.offline_dataset_server import Episode


class GymEnv(Environment):
    def __init__(self, **configs):
        super(GymEnv, self).__init__(**configs)

        env_id = self._configs["env_id"]
        scenario_configs = self._configs.get("scenario_configs", {})

        self.is_sequential = False
        self._env = gym.make(env_id, **scenario_configs)
        self._default_agent = "agent"
        self._env.possible_agents = [self._default_agent]
        self._env.observation_spaces = {
            self._default_agent: self._env.observation_space
        }
        self._env.action_spaces = {self._default_agent: self._env.action_space}
        self._trainable_agents = [self._default_agent]

    def step(self, actions: Dict[AgentID, Any]) -> Dict[str, Any]:
        observations, rewards, dones, infos = self._env.step(
            actions[self._default_agent]
        )
        agent = self._default_agent
        self._total_rewards[agent] += rewards
        self._cnt += 1

        return {
            Episode.CUR_OBS: {agent: observations},
            Episode.REWARD: {agent: rewards},
            Episode.DONE: {agent: dones},
            Episode.INFO: {agent: infos},
        }

    def render(self, *args, **kwargs):
        self._env.render()

    def reset(self):
        observations = super(GymEnv, self).reset()
        return {Episode.CUR_OBS: {self._default_agent: observations}}
