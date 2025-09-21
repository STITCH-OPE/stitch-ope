import numpy as np
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from opelab.core.mlp import MLP


class ModelScoreFunction:
    
    def __init__(self, model:MLP, learning_rate:float=1e-3, reg:float=0.) -> None:
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        
        apply_fn = self.model.apply
        
        def predict_score_fn(params, states, actions, next_states):
            x_batched = jnp.concatenate([states, actions, next_states], axis=-1)
            return jax.vmap(apply_fn, in_axes=(None, 0))(params, x_batched)
        
        def loss_fn(params, state, action, next_state):
            
            def apply_fn_next_state(x):
                x_inputs = jnp.concatenate([state, action, x], axis=-1)
                return apply_fn(params, x_inputs)
            
            score = apply_fn_next_state(next_state)
            jac = jax.jacfwd(apply_fn_next_state)(next_state)
            loss_val = jnp.trace(jac) + jnp.square(jnp.linalg.norm(score)) / 2.
            for param in jax.tree_util.tree_leaves(params):
                loss_val += reg * jnp.sum(jnp.square(param))
            return loss_val
        
        def batched_loss_fn(params, states, actions, next_states):
            losses = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0))(
                params, states, actions, next_states)
            return jnp.mean(losses)
            
        def update_fn(params, opt_state, states, actions, next_states):
            grads = jax.grad(batched_loss_fn)(params, states, actions, next_states)
            updates, next_opt_state = self.optimizer.update(grads, opt_state)
            next_params = optax.apply_updates(params, updates)
            return next_params, next_opt_state
        
        self.predict_fn = jax.jit(predict_score_fn)
        self.loss_fn = jax.jit(batched_loss_fn)
        self.update_fn = jax.jit(update_fn)
    
    def reset(self, state_dim:int, action_dim:int) -> None:
        assert (self.model.layers[-1] == state_dim)
        self.input_shape = state_dim + action_dim + state_dim
        dummy_input = np.zeros((self.input_shape,))
        self.params = self.model.init(jax.random.PRNGKey(0), dummy_input)
        self.opt_state = self.optimizer.init(self.params)
    
    def fit(self, states:np.ndarray, actions:np.ndarray, next_states:np.ndarray, 
            batch_size:int=256) -> float:
        n_samples = states.shape[0]
        perm = np.random.permutation(n_samples)
        num_full_batches, rem_batch = divmod(n_samples, batch_size)
        for i in range(num_full_batches + bool(rem_batch)):
            idx = perm[i * batch_size:(i + 1) * batch_size]
            self.params, self.opt_state = self.update_fn(
                self.params, self.opt_state, 
                states[idx], actions[idx], next_states[idx])
        loss_val = self.loss_fn(self.params, states, actions, next_states)
        return loss_val
                
    def predict(self, states:np.ndarray, actions:np.ndarray, next_states:np.ndarray) -> np.ndarray:
        return np.asarray(self.predict_fn(self.params, states, actions, next_states))
    
