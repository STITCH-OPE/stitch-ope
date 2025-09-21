from flax.training import train_state
import jax
import jax.numpy as jnp
import numpy as np
import optax

from opelab.core.mlp import MLP


class RewardEstimator:
    
    def __init__(self, mlp:MLP, learning_rate:float=1e-3, batch_size:int=64):
        self.mlp = mlp
        self.optimizer = optax.adam(learning_rate)
        self.batch_size = batch_size
    
        def predict_reward_fn(params, inputs):
            rewards = jax.vmap(self.mlp.apply, in_axes=(None, 0))(params, inputs)
            return rewards.reshape((-1, 1))
            
        def train_fn(mlp_state, inputs, rewards):

            def loss_fn(params):
                pred_rewards = predict_reward_fn(params, inputs)
                true_rewards = rewards.reshape(pred_rewards.shape)
                return jnp.mean(jnp.square(true_rewards - pred_rewards))

            loss, grads = jax.value_and_grad(loss_fn)(mlp_state.params)
            mlp_state = mlp_state.apply_gradients(grads=grads)
            return mlp_state, loss

        self.predict_fn = jax.jit(predict_reward_fn)
        self.train_fn = jax.jit(train_fn)
        self.params = None

    def fit(self, inputs, rewards, iters, seed):
        n = inputs.shape[0]
        
        # initialize model and optimizer
        mlp_state = train_state.TrainState.create(
            apply_fn=self.mlp.apply,
            params=self.mlp.init(jax.random.PRNGKey(seed), inputs[0]),
            tx=self.optimizer
        )
        
        # train reward model
        mean_loss, count_loss = 0.0, 0
        for t in range(iters):
            i = np.random.choice(n, size=self.batch_size)
            mlp_state, loss_val = self.train_fn(mlp_state, inputs[i], rewards[i])
            mean_loss = (mean_loss * count_loss + loss_val) / (count_loss + 1)
            count_loss += 1
            if t % (iters // 20) == 0:
                print(f'iter {t} mse {mean_loss}')
                mean_loss, count_loss = 0.0, 0
        self.params = mlp_state.params
    
    def predict(self, inputs):
        return self.predict_fn(self.params, inputs)


class RewardEnsembleEstimator:
    
    def __init__(self, mlp:MLP, learning_rate:float=1e-3, batch_size:int=32,
                 n_bootstraps:int=1, subsample_fraction:float=0.5):
        self.subsample_fraction = subsample_fraction
        self.bootstraps = [RewardEstimator(mlp, learning_rate, batch_size)
                           for _ in range(n_bootstraps)]
    
    def fit(self, inputs, rewards, iters, seeds):
        n = inputs.shape[0]
        n_sample = int(self.subsample_fraction * n)
        for model, seed in zip(self.bootstraps, seeds):
            i = np.random.choice(n, size=n_sample)
            model.fit(inputs[i], rewards[i], iters, seed)
    
    def predict(self, inputs):
        predictions = [model.predict(inputs).reshape((-1,)) 
                       for model in self.bootstraps]
        return np.stack(predictions, axis=-1)
    

if __name__ == '__main__': 
    
    def reward_fn(states):
        return np.sin(states * 2 * np.pi)
    
    # train
    states = np.random.uniform(low=-1, high=1, size=(50, 1))
    rewards = reward_fn(states)
    mlp = MLP([32, 32, 1])
    est = RewardEnsembleEstimator(mlp)
    est.fit(states, rewards, 10000, [0, 1, 2, 3, 4])
    
    # test
    states_test = np.random.uniform(low=-2, high=2, size=(500, 1))
    states_test = np.sort(states_test, axis=0)
    rewards_test = reward_fn(states_test)
    est_rewards = est.predict(states_test)
    mean_rewards = np.mean(est_rewards, axis=-1)
    std_rewards = np.std(est_rewards, axis=-1)
    states_test = states_test.reshape((-1,))
    
    # plot
    import matplotlib.pyplot as plt
    plt.plot(states_test, rewards_test, label='truth')
    plt.plot(states_test, mean_rewards, label='estimate')
    plt.fill_between(states_test, mean_rewards - 2 * std_rewards, mean_rewards + 2 * std_rewards, alpha=0.5)
    plt.legend()
    plt.show()
