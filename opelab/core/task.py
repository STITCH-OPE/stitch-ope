import gym
from gym.envs.classic_control import AcrobotEnv
from gym import Wrapper
import numpy as np
from gym import spaces
from numpy import pi, cos


class ContinuingTask(gym.Env):
    
    def __init__(self, base:gym.Env, reset_reward:float) -> None:
        super(ContinuingTask, self).__init__()
        self.base = base
        self.reset_reward = reset_reward
        self.action_space = base.action_space
        self.observation_space = base.observation_space
        
    def reset(self, seed=None, options=None):
        return self.base.reset(seed=seed, options=options)
    
    def step(self, action):
        state, reward, terminated, truncated, _ = self.base.step(action)
        if terminated:
            state, _ = self.base.reset()
            reward = self.reset_reward
            terminated = False
            truncated = False
        return state, reward, terminated, truncated, {}
    
    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()


class TerminationTask(gym.Env):
    
    def __init__(self, base:gym.Env, reset_reward:float) -> None:
        super(TerminationTask, self).__init__()
        self.base = base
        self.reset_reward = reset_reward
        self.action_space = base.action_space
        self.observation_space = base.observation_space
        
    def reset(self, seed=None, options=None):
        return self.base.reset(seed=seed, options=options)
    
    def step(self, action):
        state, reward, terminated, truncated, _ = self.base.step(action)
        if terminated:
            reward = self.reset_reward
        return state, reward, terminated, truncated, {}
    
    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()


class ContinuingTaskTwo(gym.Env):
    
    def __init__(self, base:gym.Env, reset_reward:float) -> None:
        super(ContinuingTaskTwo, self).__init__()
        self.base = base
        self.action_space = base.action_space
        self.observation_space = base.observation_space
        self.ended = False
        self.reset_reward = reset_reward
        
    def reset(self, seed=None, options=None):
        self.ended = False
        return self.base.reset(seed=seed, options=options)
    
    def step(self, action):
        state, reward, terminated, truncated, _ = self.base.step(action)

        # If the agent has finished, it will motivate it to stay at the finish state and get the end reward
        if self.ended and terminated:
            terminated = False
            truncated = False
            return state , self.reset_reward, terminated, truncated, {}

        if terminated:
            terminated = False
            truncated = False
            self.ended = True

        return state, reward, terminated, truncated, {}
    
    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()


class ContinuingTaskThree(gym.Env):
    
    def __init__(self, base:gym.Env, terminal_reward:float) -> None:
        super(ContinuingTaskThree, self).__init__()
        self.base = base
        self.action_space = base.action_space
        self.observation_space = base.observation_space
        self.terminal_reward = terminal_reward
        
    def reset(self, seed=None, options=None):
        return self.base.reset(seed=seed, options=options)
    
    def step(self, action):
        state, reward, terminated, truncated, _ = self.base.step(action)

        # If the agent has terminated, it receives the terminal_reward and continues on
        if terminated:
            reward = self.terminal_reward
            terminated = False

        return state, reward, terminated, truncated, {}
    
    def render(self):
        return self.base.render()
    
    def close(self):
        self.base.close()

    
class ContinuingTaskAcrobat(gym.Env):

    def __init__(self, base:gym.Env, horizon:int, reward=5) -> None:
        super(ContinuingTaskAcrobat, self).__init__()
        self.base = base
        self.action_space = base.action_space
        self.observation_space = base.observation_space
        self.ended = False
        self.horizon = horizon
        self.reward = reward

    def reset(self, seed=None, options=None):
        self.ended = False
        self.timestep = 0
        return self.base.reset(seed=seed, options=options)
    
    def step(self, action):
        state, reward, terminated, truncated, _ = self.base.step(action)
        self.timestep += 1

        if terminated:
            terminated = False
            truncated = False
            reward = self.reward
        
        if self.timestep > self.horizon:
            terminated = True
            truncated = True

        return state, reward, terminated, truncated, {}
    
    def render(self):
        return self.base.render()

    def close(self):
        self.base.close()

class ContinuousAcrobotEnv(AcrobotEnv):
    """
    Infinite Horizon Acrobot with success rewards
    - Continuous action space: [-1.0, 1.0]
    - Reward: +100 when vertical height > 1.0 (original success condition)
    - Never terminates (infinite horizon)
    - Default truncation at 500 steps from TimeLimit wrapper
    """
    
    def __init__(self):
        super().__init__()
        
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )
        
        self._max_episode_steps = 500
        self.spec = gym.envs.registration.EnvSpec("ContinuousAcrobot")
        

    def step(self, action):
        # Original dynamics calculation
        torque = np.clip(action[0], -1.0, 1.0)
        
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        s_augmented = np.append(self.state, torque)
        ns = self.rk4(self._dsdt, s_augmented, [0, self.dt])[:4]

        ns[0] = self.wrap(ns[0], -pi, pi)
        ns[1] = self.wrap(ns[1], -pi, pi)
        ns[2] = self.bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = self.bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        
        self.state = ns

        theta1 = self.state[0]
        theta2 = self.state[1]
        success = (-cos(theta1) - cos(theta1 + theta2)) > 1.0
        reward = 10.0 if success else -1.0
        
        # With shaped reward guidance:
        # theta1 = self.state[0]
        # current_height = -cos(theta1) - cos(theta1 + self.state[1])
        # height_reward = current_height * 10  # Scale height to [-20, 20]
        # success_bonus = 100.0 if current_height > 1.0 else 0.0
        # energy_penalty = 0.01 * np.square(action[0])  # Penalize large torques
        # reward = height_reward + success_bonus - energy_penalty

        terminated = False
        self.t += 1
                
        return self._get_ob(), reward, terminated, False

    def _dsdt(self, s_augmented):
        derivatives = super()._dsdt(s_augmented)
        return np.array(derivatives, dtype=np.float32)
    
    def reset(self, *, seed=None, options = None):
        obs = super().reset(seed=seed, options=options)
        self.t = 0
        return obs

    def _terminal(self):
        terminated = super()._terminal() or self.t >= self._max_episode_steps
        return terminated
        
    @staticmethod
    def wrap(x, m, M):
        """Angle wrapping"""
        diff = M - m
        while x > M: x -= diff
        while x < m: x += diff
        return x

    @staticmethod
    def bound(x, m, M=None):
        """Velocity bounding"""
        return np.clip(x, m, M) if M else np.clip(x, m[0], m[1])

    def rk4(self, derivs, y0, t):
        """Fixed RK4 implementation with proper array handling"""
        y0 = np.asarray(y0, dtype=np.float32)
        yout = np.zeros((len(t), len(y0)), dtype=np.float32)
        yout[0] = y0
        
        for i in range(len(t)-1):
            dt = t[i+1] - t[i]
            dt2 = dt/2.0
            
            k1 = derivs(yout[i])
            k2 = derivs(yout[i] + dt2 * k1)
            k3 = derivs(yout[i] + dt2 * k2)
            k4 = derivs(yout[i] + dt * k3)
            
            yout[i+1] = yout[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
            
        return yout[-1]
