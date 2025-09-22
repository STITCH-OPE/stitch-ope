import numpy as np
from flax.training import train_state, checkpoints
import jax
import jax.numpy as jnp
import optax
from typing import Any, AnyStr, Callable, Sequence, Tuple

from opelab.core.mlp import MLP


class Replay:
    
    def __init__(self, n_samples:int=50000, n_batch:int=32) -> None:
        self.n_samples = n_samples
        self.n_batch = n_batch
    
    def reset(self) -> None:
        self.buffer = np.empty(self.n_samples, dtype=object)
        self.index = 0
        self.size = 0
    
    def replay(self) -> Tuple[np.ndarray, ...]:
        if self.size < self.n_batch: 
            return None
        indices = np.random.randint(low=0, high=self.size, size=(self.n_batch,))
        states, actions, rewards, next_states, gammas = zip(*self.buffer[indices])
        states = np.vstack(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.vstack(next_states)
        gammas = np.array(gammas)
        return states, actions, rewards, next_states, gammas
    
    def append(self, state:Any, action:Any, reward:float, next_state:Any, gamma:float) -> None:
        self.buffer[self.index] = (state, action, reward, next_state, gamma)
        self.size = min(self.size + 1, self.n_samples)
        self.index = (self.index + 1) % self.n_samples


class QLearning:
    
    def __init__(self, n_state:int, n_action:int, alpha:float, gamma:float=1.0,
                 chkp=None) -> None:
        self.n_state = n_state
        self.n_action = n_action
        self.alpha = alpha
        self.gamma = gamma
        if chkpt is None:
            self.Q = np.random.rand(n_state, n_action)
        else:
            self.Q = np.load(chkpt)

    def update(self, s:int, a:int, r:float, s1:int, done:bool) -> float:
        gamma = 0 if done else self.gamma
        y = r + gamma * np.max(self.Q[s1])
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * y
        loss = abs(self.Q[s, a] - y)
        return loss
    
    def values(self, state:int) -> np.ndarray:
        return self.Q[state]


class DQN:
    
    def __init__(self, layers:Sequence[int], dummy_state:np.ndarray, opt=optax.adam(0.001),
                 gamma:float=1.0, n_target_update:int=10, key:int=42, replay:Replay=Replay(),
                 chkpt=None) -> None:
        self.optimizer = opt
        self.gamma = gamma
        self.n_target_update = n_target_update
        self.replay = replay
        self.replay.reset()
        
        model = MLP(layers)
        self.apply_fn = model.apply
        self.params = model.init(jax.random.PRNGKey(key), dummy_state)
        self.opt_state = self.optimizer.init(self.params)
        self.target_params, self.counter = self.params, 0
                
        if chkpt is not None: 
            state = train_state.TrainState.create(
                apply_fn=self.apply_fn, params=self.params, tx=self.optimizer)
            restored_state = checkpoints.restore_checkpoint(ckpt_dir=chkpt, target=state)
            self.params = self.target_params = restored_state.params
        self._compile()
        
    def _compile(self):
        
        def predict_fn(params, states):
            return jax.vmap(self.apply_fn, in_axes=(None, 0))(params, states)
        
        def target_fn(reward, next_q, next_action, gamma):
            return reward + gamma * next_q[next_action]
        
        def current_fn(current_q, action):
            return current_q[action]
        
        def loss_fn(params, target_params, transitions):
            states, actions, rewards, next_states, next_actions, gammas = transitions
            current_q = predict_fn(params, states)
            current_q = jax.vmap(current_fn)(current_q, actions)
            next_q = predict_fn(target_params, next_states)
            target_q = jax.vmap(target_fn)(rewards, next_q, next_actions, gammas)
            target_q = jax.lax.stop_gradient(target_q)
            return jnp.mean(jnp.square(target_q - current_q))

        def update_fn(params, target_params, transitions, opt_state):
            loss, grads = jax.value_and_grad(loss_fn)(params, target_params, transitions)
            updates, next_opt_state = self.optimizer.update(grads, opt_state)
            next_params = optax.apply_updates(params, updates)
            return next_params, next_opt_state, loss
        
        self.predict_fn = jax.jit(predict_fn)
        self.update_fn = jax.jit(update_fn)        
    
    def update(self, s:np.ndarray, a:int, r:float, s1:np.ndarray, done:bool) -> float:
        loss = np.nan
        self.replay.append(s, a, r, s1, 0.0 if done else self.gamma)
        batch = self.replay.replay()
        if batch is not None:
            states, actions, rewards, next_states, gammas = batch
            next_q = np.asarray(self.predict_fn(self.params, next_states))
            next_actions = np.argmax(next_q, axis=1)
            transitions = (states, actions, rewards, next_states, next_actions, gammas)
            self.params, self.opt_state, loss = self.update_fn(
                self.params, self.target_params, transitions, self.opt_state)
            self.counter += 1
            if self.counter >= self.n_target_update:
                self.target_params, self.counter = self.params, 0
        return loss
    
    def values(self, state:np.ndarray) -> np.ndarray:
        state = state.reshape((1, -1))
        return np.asarray(self.predict_fn(self.params, state)).reshape((-1,))
    
    def save(self, path:AnyStr, step:int=0) -> None:
        state = train_state.TrainState.create(
            apply_fn=self.apply_fn, params=self.params, tx=self.optimizer)
        checkpoints.save_checkpoint(ckpt_dir=path, target=state, step=step)
