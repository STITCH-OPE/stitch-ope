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
from copy import deepcopy
import opelab.core.baselines.diffusion.utils as utils
from opelab.core.baselines.diffusion.helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses,
)

from opelab.core.policy import DiffusionPolicy

torch.autograd.set_detect_anomaly(True)

Sample = namedtuple('Sample', 'trajectories values chains')
f = open('diffuser.txt', 'w')
print("timestep\ttarget_likelihood\tbehaviour_likelihood", file=f, flush=True)


def gradlog(policy, 
            x, 
            state_dim, 
            action_dim, 
            normalize=True, 
            gmode=False, 
            verbose=False,
            reverse=False):
  
    N, T, D = x.shape
    assert D == state_dim + action_dim, "Inconsistent dimensions"

    x_detached = x.detach()     
    if not reverse:        
        state_t = x_detached[:, :, :state_dim].clone().requires_grad_(False)
        action_t = x_detached[:, :, state_dim:].clone().requires_grad_(True)
    else:
        state_t = x_detached[:, :, action_dim:].clone().requires_grad_(False)
        action_t = x_detached[:, :, :action_dim].clone().requires_grad_(True)
        
    
    x_for_grad = torch.cat([state_t, action_t], dim=2)  

    state_t_flat = x_for_grad[:, :, :state_dim].reshape(-1, state_dim)
    action_t_flat = x_for_grad[:, :, state_dim:].reshape(-1, action_dim)

    if gmode:
        log_prob = policy.gaussian_log_prob(state_t_flat, action_t_flat)
    else:
        log_prob = policy.log_prob_extended(state_t_flat, action_t_flat)

    log_prob = log_prob.sum(dim=1)

    log_prob = log_prob.view(N, T)              
    log_likelihood = log_prob.sum(dim=1)        
    total_log_likelihood = log_likelihood.sum() 

    if verbose:
        print('Total Log Likelihood:', total_log_likelihood.item())

    grad_action = torch.autograd.grad(
        total_log_likelihood,        
        action_t,                      
        retain_graph=False, 
        create_graph=False
    )[0]  

    if normalize:
        epsilon = 1e-6
        norm_per_timestep = grad_action.norm(dim=-1, keepdim=True) + epsilon
        grad_action = grad_action / norm_per_timestep 

    grad_log_likelihood = torch.zeros_like(x_detached) 
    if not reverse:
        grad_log_likelihood[:, :, state_dim:] = grad_action
    else:
        grad_log_likelihood[:, :, :action_dim] = grad_action

    return grad_log_likelihood

def gradlog_diffusion(policy, x, state_dim, action_dim, normalize=False):
    N, T, D = x.shape
    states = x[:, :, :state_dim]
    actions = x[:, :, state_dim:]
    grad_log_prob = torch.zeros_like(x)
    # flatten NxTxD to (N*T)xD no requires_grad
    states = states.reshape(-1, state_dim)
    actions = actions.reshape(-1, action_dim)
    grad_action = policy.grad_log_prob(states, actions)
    grad_action = grad_action.reshape(N, T, action_dim)
    
    if normalize:
        norm = torch.norm(grad_action, dim=-1, keepdim=True) + 1e-6 
        grad_action = grad_action / norm
        
    # change the shape back to NxTxD
    grad_log_prob[:, :, :state_dim] = 0
    grad_log_prob[:, :, state_dim:] = grad_action
    
    return grad_log_prob

def get_schedule_multiplier(t, n_timesteps, schedule_type='cosine'):
    if isinstance(t, torch.Tensor):
        t = t.item()
    t_frac = t / n_timesteps
    
    if schedule_type == 'cosine':
        return 0.5 * (1 + math.cos(math.pi * t_frac))
    
    elif schedule_type == 'linear':
        return 1 - t_frac
    
    elif schedule_type == 'sigmoid':
        k = 10  
        mid = 0.5  
        return 1 / (1 + math.exp(k * (t_frac - mid)))
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
@torch.no_grad
def log(policy, behavior_policy, x, t, f, state_dim):
    with torch.no_grad():
        all_logs_target = 0
        all_logs_behavior = 0
        x_clone = x.clone()
        for batch in range(x.shape[0]):
            state_t = x_clone[batch, :, :state_dim].detach()
            action_t = x_clone[batch, :, state_dim:].detach()
            
            log_prob = policy.log_prob(state_t, action_t)
            log_likehlihood = log_prob.sum()
            all_logs_target += log_likehlihood.item() / x.shape[1]
            
            log_prob = behavior_policy.log_prob(state_t, action_t)
            log_likehlihood = log_prob.sum()
            all_logs_behavior += log_likehlihood.item() / x_clone.shape[1]
            
        print(f"{t}\t{all_logs_target / x.shape[0]}\t{all_logs_behavior / x.shape[0]}\t", file=f, flush=True)   


def default_sample_fn(model, x, t, state_dim, action_dim, guided, guidance_hyperparams, gmode= False, policy=None, behavior_policy=None, transform=None, unnormalizer=None, normalizer=None, cond=None, return_info=False, reverse=False):
    
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        model_mean, _, model_log_variance = model.p_mean_variance(x=x, t=t)
        model_std = torch.exp(0.5 * model_log_variance)
        
    k = guidance_hyperparams['k_guide']
    normalize_grad = guidance_hyperparams['normalize_grad']
    use_adaptive = guidance_hyperparams['use_adaptive']
    use_neg_grad = guidance_hyperparams['use_neg_grad']
    neg_grad_scale = guidance_hyperparams['neg_grad_scale']
    action_scale = guidance_hyperparams['action_scale']
    state_scale = guidance_hyperparams['state_scale']
    use_action_grad_only = guidance_hyperparams['use_action_grad_only']
    clamp = guidance_hyperparams['clamp']
    l_inf = guidance_hyperparams['l_inf']
    ratio = guidance_hyperparams['ratio']
        
        
    normalize_v = not clamp and normalize_grad
    
    info = [None, None]
    
    if return_info:
        if use_action_grad_only:
            mmp = apply_conditioning(model_mean, cond, state_dim, reverse=reverse)
            info[0] = (mmp-x)[..., state_dim:]
                
    model_mean = unnormalizer(model_mean)    
    if guided:
        assert policy is not None 
        if use_neg_grad:
            assert behavior_policy is not None
        
        gmode=gmode
        # print('gmode: ', gmode)
        
        for _ in range(k):
            if policy.__class__.__name__ == 'DiffusionPolicy':
                gradient = gradlog_diffusion(policy, model_mean, state_dim, action_dim, normalize=normalize_v)
            else:
                gradient = gradlog(policy, model_mean, state_dim, action_dim, normalize=normalize_v , gmode=gmode, verbose=False, reverse=reverse)
            neg_grad = 0
            if use_neg_grad:
                neg_grad = gradlog(behavior_policy, model_mean, state_dim, action_dim, normalize=normalize_v, gmode=gmode, verbose=False, reverse=reverse)
            
            if use_neg_grad:   
                # print('Action: ', model_mean[..., state_dim:])
                # print('Gradient: ', gradient[..., state_dim:])
                # print('Neg Gradient: ', neg_grad[..., state_dim:])
                
                if normalize_grad:
                    
                    if clamp:
                        gradient = torch.clamp(gradient, min=-l_inf, max=l_inf)
                        neg_grad = torch.clamp(ratio * neg_grad, min=-l_inf, max=l_inf)
                    else:
                        neg_grad = ratio * neg_grad
                        
                guide = gradient - neg_grad
                    
                # guide = torch.clamp(guide, min=-1, max=1)                             
                
            else:
                if normalize_grad and clamp:
                    gradient = torch.clamp(gradient, min=-l_inf, max=l_inf)
                    
                guide = gradient
                #print('Gradient: ', gradient[..., state_dim:])
                # guide = torch.clamp(guide, min=-1, max=1) 
            
            if not use_adaptive:
                guide = 1 * action_scale * guide
            else:
                #guide = model_std * action_scale * guide
                scale_muliplier = get_schedule_multiplier(model.n_timesteps - t[0].item(), model.n_timesteps, schedule_type='cosine')
                #print('model_std: ', model_std[0].item(), 'scale_multiplier: ', scale_muliplier)
                guide = scale_muliplier * action_scale * guide 
           
            #print('Guide: ', guide[..., state_dim:])

            model_mean = model_mean + guide            
            
            model_mean = normalizer(model_mean)
            model_mean = apply_conditioning(model_mean, cond, state_dim, reverse=reverse)
            model_mean = unnormalizer(model_mean)

            if return_info:
                info[1] = guide[..., state_dim:]
        
    model_mean = normalizer(model_mean)

    with torch.no_grad():
        noise = torch.randn_like(x)
        noise[t == 0] = 0  
            
    return model_mean + model_std * noise, info


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l2', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None, policy=None, behavior_policy=None, normalizer=None, unnormalizer=None, transform=None, guided=True, gmode=False, reverse=False
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
        self.transform = transform
        self.guided = guided
        self.gmode = gmode
        self.reverse = reverse

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
            dim_weights[ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## manually set a0 weight
        loss_weights[0, self.observation_dim:] = action_weight
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
            x_recon.clamp_(-5., 5.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, guided=False, guidance_hyperparams=None, return_info = False, **sample_kwargs):
        device = self.betas.device
        with torch.no_grad():
            batch_size = shape[0]
            x = torch.randn(shape).to(device=device)

            chain = [x] if return_chain else None

        if cond is not None:
            x = apply_conditioning(x, cond, self.observation_dim, reverse=self.reverse)


        guidance_grads_over_time = []
        model_preds_over_time = []

        progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            #print(i)
            if self.policy is not None:
                sample_kwargs['policy'] = self.policy
                sample_kwargs['behavior_policy'] = self.behavior_policy
            
            if cond is not None:
                sample_kwargs['cond'] = cond
            if self.transform is not None:
                sample_kwargs['transform'] = self.transform
            if self.normalizer is not None:
                sample_kwargs['normalizer'] = self.normalizer
                sample_kwargs['unnormalizer'] = self.unnormalizer
            else:
                print('Warning! No normalization is being used')
                
            x, info = sample_fn(self, x, t, state_dim=self.observation_dim, action_dim=self.action_dim, guided=guided, guidance_hyperparams=guidance_hyperparams,return_info=return_info, gmode=self.gmode, reverse=self.reverse, **sample_kwargs)
            if return_info:
                model_preds_over_time.append(info[0].detach().cpu().numpy().copy())
                guidance_grads_over_time.append(info[1].detach().cpu().numpy().copy())
            if cond is not None:
                x = apply_conditioning(x, cond, self.observation_dim, reverse=self.reverse)

            progress.update({'t': i})
            if return_chain: chain.append(x)
        progress.stamp()

        if return_info:
            info = {'model_predictions': np.stack(model_preds_over_time), 'guidance': np.stack(guidance_grads_over_time)}

        if return_chain: chain = torch.stack(chain, dim=1)
        x = x.detach()
        if return_info:
            return Sample(x, None , chain), info
        else:
            return Sample(x, None , chain)
    #@torch.no_grad()
    def conditional_sample(self, shape, cond, verbose=True, return_chain=False, action_scale=0.2, state_scale=0.01, guided=False, use_adaptive=True, use_neg_grad=True, neg_grad_scale=0.1, normalize_grad=True, k_guide=2, use_action_grad_only=True,return_info=False, clamp=False, l_inf=1, ratio=1, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        guidance_hyperparams = {
            'action_scale': action_scale,
            'state_scale': state_scale,
            'use_adaptive': use_adaptive,
            'use_neg_grad': use_neg_grad,
            'neg_grad_scale': neg_grad_scale,
            'normalize_grad': normalize_grad,
            'k_guide': k_guide,
            'use_action_grad_only': use_action_grad_only,
            'l_inf': l_inf,
            'ratio': ratio,
            'clamp': clamp,
        }
        
        device = self.betas.device
        batch_size = shape[0]
        horizon = shape[1]
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, guided=guided, guidance_hyperparams=guidance_hyperparams, return_info=return_info, **sample_kwargs)

    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, t, cond):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = apply_conditioning(x_noisy, cond, self.observation_dim, reverse=self.reverse)
                
        x_recon = self.model(x_noisy, t)
        x_recon = apply_conditioning(x_recon, cond, self.observation_dim, reverse=self.reverse)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss, info = self.loss_fn(x_recon, noise)
        else:
            loss, info = self.loss_fn(x_recon, x_start)

        return loss, info

    def loss(self, x, cond):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, t, cond)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)
