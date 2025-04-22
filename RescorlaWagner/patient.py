import gymnasium as gym
import numpy as np

# delta_v_t = alpha * beta (lambda * US_t - V_t) * CS_t
CS_COUNT = 3
MIN_V_APPROX = 0.08

class Actions:
    CS_1 = 0
    CS_2 = 1
    CS_3 = 2
    NO_CS = 3

    def is_CS(action):
        return action in [Actions.CS_1, Actions.CS_2, Actions.CS_3]
    
    def to_str(action):
        if action == Actions.CS_1:
            return "CS_low"
        elif action == Actions.CS_2:
            return "CS_mid"
        elif action == Actions.CS_3:
            return "CS_high"
        else:
            return "NO_CS"


class Patient(gym.Env):
    def __init__(self, V, alpha_x, cost_x, cost_threshold, beta=1, max_US=1, max_trials=50):
        super(Patient, self).__init__()
        self.V = V
        self.beta = beta
        self.max_US = max_US
        self.max_trials = max_trials
        self.alpha_x = alpha_x
        self.cost_x = cost_x
        self.cost_threshold = cost_threshold

        self.streak = 0
        self.cost = 0
        self.trials = 0
        
        self.terminated = False
        self.truncated = False
        
        self.action_space = gym.spaces.Discrete(CS_COUNT + 1)
        self.observation_space = gym.spaces.Box(
            low=-1000, high=1000, shape=(len(self.__get_obs()),), dtype=float)

    def step(self, action):
        if self.terminated or self.truncated:
            raise Exception("Environment is already terminated or truncated.")

        old_v = self.V
        self.trials += 1
        delta = self.max_US * self._get_US_t() - self.V
        if Actions.is_CS(action):
            self.streak += 1
            self.V += self.alpha_x[action] * self.beta * delta
            self.cost += self.__get_cost(action)
        else:
            self.streak = 0

        if self.V == 0:
            self.terminated = True

        if self.trials >= self.max_trials:
            self.truncated = True

        if self.cost >= self.cost_threshold:
            self.terminated = True

        reward = self.__get_reward(action, old_v)
        if self.cost >= self.cost_threshold:
            reward -= 10
            self.terminated = True

        return self.__get_obs(), reward, self.terminated, self.truncated, {"cost": self.cost}

    def reset(self):
        self.trials = 0
        self.streak = 0
        self.V = 1
        self.cost = 0
        self.terminated = False
        self.truncated = False
        return self.__get_obs(), {}

    def render(self, mode='ansi'):
        print(f"Trial: {self.trials}, V: {self.V} C: {self.cost}")
        pass

    def __get_obs(self):
        return {"V": self.V, "cost": self.cost, "trials": self.trials}

    def __get_reward(self, action, old_v):
        # return (old_v-self.V) - (self.__get_cost(action) / self.cost_threshold)
        return (old_v-self.V)
        # return -self.V

    def __get_cost(self, action):
        if not Actions.is_CS(action):
            return 0
        else:
            base_cost = self.cost_x[action]
            streak_factor = 1 + 0.25*(self.streak-1)
            return base_cost * streak_factor

    # Extinction
    def _get_US_t(self):
        return 0
