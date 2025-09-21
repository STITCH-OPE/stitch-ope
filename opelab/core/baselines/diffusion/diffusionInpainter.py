from collections import namedtuple
import gc
import numpy as np
import pdb
import sys
import os
import copy 
import math

import torch
from torch import nn
from torch.autograd import grad

import opelab.core.baselines.diffusion.utils as utils
from opelab.core.baselines.diffusion.helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)


torch.autograd.set_detect_anomaly(True)

Sample = namedtuple('Sample', 'trajectories values chains')
f = open('inpainter.txt', 'w')
print("timestep\ttarget_likelihood\tbehaviour_likelihood", file=f, flush=True)

def inpaint(policy, x, state_dim):
    batch_size, horizon, total_dim = x.shape
    action_dim = total_dim - state_dim
    states = x[:, :, :state_dim]  
    actions = x[:, :, state_dim:] 
    new_actions = []
    epsilon = 1e-5
    for t in range(horizon):
        state_t = states[:, t, :]  
        action_t = 0.5 * policy.sample(state_t.cpu())
        action_t = torch.tensor(np.arctanh(action_t.clip(-1 + epsilon , 1 - epsilon)),device=x.device)
        new_actions.append(action_t)
    new_actions = torch.stack(new_actions, dim=1)  
    new_tensor = torch.cat((states, new_actions), dim=-1)
    return new_tensor


@torch.no_grad
def log(policy, behavior_policy, x, t, f, state_dim):
    with torch.no_grad():
        all_logs_target = 0
        all_logs_behavior = 0
        x_clone = x.clone()
        for batch in range(x.shape[0]):
            state_t = x_clone[batch, :, :state_dim].detach()
            action_t = x_clone[batch, :, state_dim:].detach()
            #action_t = torch.clamp(action_t, min=-2, max=2)
            
            action_t = 2 * torch.tanh(action_t)
            
            log_prob = policy.log_prob(state_t, action_t)
            log_likehlihood = log_prob.sum()
            all_logs_target += log_likehlihood.item() / x.shape[1]
            
            log_prob = behavior_policy.log_prob(state_t, action_t)
            log_likehlihood = log_prob.sum()
            all_logs_behavior += log_likehlihood.item() / x_clone.shape[1]

            #log_prob = policy.gaussian_log_prob(state_t, action_t)
            #log_likehlihood = log_prob.sum()
            #all_logs_target += log_likehlihood.item() / x.shape[1]
            #log_prob = behavior_policy.gaussian_log_prob(state_t, action_t)
            #log_likehlihood = log_prob.sum()
            #all_logs_behavior += log_likehlihood.item() / x.shape[1]

        print(f"{t}\t{all_logs_target / x.shape[0]}\t{all_logs_behavior / x.shape[0]}\t", file=f, flush=True)   

def default_sample_fn(model, x, t,state_dim, action_dim, scale=0.01, policy=None, behavior_policy=None, unnormalizer=None, normalizer=None, start=256, every=1):
    with torch.no_grad():
        model_log_variance = extract(model.posterior_log_variance_clipped, t, x.shape)
        model_std = torch.exp(0.5 * model_log_variance)
        model_var = torch.exp(model_log_variance)
        x = unnormalizer(x)
    if policy is not None:
        if t[0] > start and t[0] % every == 0:
            x = inpaint(policy, x, state_dim)

    if policy is not None and behavior_policy is not None:
        log(policy, behavior_policy, x, t[0].item() ,f, state_dim)
        pass

    x = normalizer(x)

    with torch.no_grad():
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, t=t)
        noise = torch.randn_like(x)
        noise[t == 0] = 0

    return model_mean + model_std * noise, None


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusionInpainter(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, policy=None, behavior_policy=None, normalizer=None, unnormalizer=None, start=256, every=1
    ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model
        self.policy = policy
        self.behavior_policy = behavior_policy
        self.normalizer = normalizer
        self.unnormalizer = unnormalizer
        self.start = start
        self.every = every

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim)

    def get_loss_weights(self, action_weight, discount, weights_dict):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, :self.action_dim] = action_weight
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t))

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_loop(self, shape, verbose=True, return_chain=False, sample_fn=default_sample_fn, scale=0, **sample_kwargs):
        device = self.betas.device
        with torch.no_grad():
            batch_size = shape[0]
            x = torch.randn(shape, device=device)

            chain = [x] if return_chain else None

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            #print(i)
            if self.policy is not None:
                sample_kwargs['policy'] = self.policy
                sample_kwargs['scale'] = scale
                sample_kwargs['behavior_policy'] = self.behavior_policy
            
            if self.normalizer is not None:
                sample_kwargs['normalizer'] = self.normalizer
                sample_kwargs['unnormalizer'] = self.unnormalizer
            else:
                print('Warning! No normalization is being used')
            
            sample_kwargs['start'] = self.start
            sample_kwargs['every'] = self.every
                
            x, _ = sample_fn(self, x, t, state_dim=self.observation_dim, action_dim=self.action_dim, **sample_kwargs)
            progress.update({'t': i})
            if return_chain: chain.append(x)
        progress.stamp()

        if self.policy is not None:
            hebz = []
            x = self.unnormalizer(x)
            for i in range(len(x)):
                s = x[i, :, :self.observation_dim]
                a = x[i, :, self.observation_dim:]
                #HARDCODE
                a = 2 * torch.tanh(a)
                
                log_prob = self.policy.log_prob(s,a)
                log_likehlihood = log_prob.sum() / (self.horizon)
                if log_likehlihood.item() > -20:
                    hebz.append(log_likehlihood.item())
                    
            print('Target Likelihood: ', np.mean(hebz))
            print(len(hebz))

            x = self.normalizer(x)
        
        if self.behavior_policy is not None:
            hebz = []
            x = self.unnormalizer(x)
            for i in range(len(x)):
                s = x[i, :, :self.observation_dim]
                a = x[i, :, self.observation_dim:]
                #HARDCODE
                a = 2 * torch.tanh(a)
                
                log_prob = self.behavior_policy.log_prob(s,a)
                log_likehlihood = log_prob.sum() / (self.horizon)
                if log_likehlihood.item() > -20:
                    hebz.append(log_likehlihood.item())
            print('Behaviour Likelihood: ', np.mean(hebz))
            print(len(hebz))
            x = self.normalizer(x)

        if self.policy is not None and self.behavior_policy is not None:
            x = self.unnormalizer(x)
            log(self.policy, self.behavior_policy, x, -1 ,f, self.observation_dim)
            x = self.normalizer(x)

        if return_chain: chain = torch.stack(chain, dim=1)
        return Sample(x, None , chain)


    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

