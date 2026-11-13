"""Helper utilities for stochastic sampling and Gaussian-process modeling.

This module provides probability-distribution wrappers for deterministic,
uniform, beta, and normal variables, as well as containers for independent
stochastic and design-variable distributions. The distribution classes
support sampling, probability-density evaluation, bounds estimation, and
joint-density calculations.

The module also provides utilities for constructing and copying BoTorch
Gaussian-process models, fitting their hyperparameters, and generating
Matheron posterior sample paths for uncertainty quantification.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

import torch
import copy, time
from botorch.models.transforms.outcome import Standardize
from botorch.models import SingleTaskGP
from gpytorch.means.constant_mean import ConstantMean
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch.distributions as dist
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths

##############################
# Stochastic sampling helper #
##############################

class DiracDist:
    def __init__(self):
        pass

    def sample(self, n_samples, device, dtype):
        sample = torch.zeros(n_samples, device = device, dtype = dtype)
        return sample

    def pdf(self, x):
        return 1
    
class UniformDist:
    def __init__(self, low, high) -> None:
        self.low = low
        self.high = high

    def get_bounds(self):
        '''Get the bounds of the uniform distribution'''
        return [self.low, self.high]

    def sample(self, n_samples, device, dtype):
        sample = self.low + (self.high - self.low) * torch.rand(n_samples, device = device, dtype = dtype)
        return sample

    def pdf(self, x):
        
        return 1/(self.high - self.low)
    
class BetaDist:
    def __init__(self, lb, ub, alpha, beta) -> None:
        self.lb = lb
        self.ub = ub
        self.alpha = alpha 
        self.beta = beta

    def get_bounds(self):
        '''Get the bounds of the beta distribution'''
        return [self.lb, self.ub]

    def sample(self, n_samples, device, dtype):
        alpha_tensor = torch.tensor(self.alpha, device=device, dtype= dtype)
        beta_tensor = torch.tensor(self.beta, device=device, dtype= dtype)
        beta_dist = torch.distributions.Beta(alpha_tensor, beta_tensor)
        samples = beta_dist.sample((n_samples,)).to(dtype).to(device)
        transformed_samples = self.lb + (self.ub - self.lb) * samples
        return transformed_samples

    def pdf(self, x):
        x = torch.as_tensor(x)
        dtype = x.dtype
        device = x.device

        lb = torch.tensor(self.lb, dtype=dtype, device=device)
        ub = torch.tensor(self.ub, dtype=dtype, device=device)
        alpha = torch.tensor(self.alpha, dtype=dtype, device=device)
        beta = torch.tensor(self.beta, dtype=dtype, device=device)

        if x < lb or x > ub: raise(ValueError("The requested x value is not in the bounds"))

        y = (x - lb) / (ub - lb)
        beta_dist = torch.distributions.Beta(alpha, beta)
        pdf = torch.exp(beta_dist.log_prob(y)) / (ub - lb)

        return pdf
    
class NormalDist:
    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def sample(self, n_samples, device, dtype):
        sample = torch.normal(self.mean, self.std, size=(n_samples,), device = device, dtype = dtype)
        return sample
    
    def get_bounds(self):
        '''Use the 99% and 1% quantile as approximate bounds of the normal distribution'''
        x_lower = self.icdf(torch.tensor(0.01)).item()
        x_upper = self.icdf(torch.tensor(0.99)).item()
        return [x_lower, x_upper]

    def icdf(self, x):
        normal_dist = dist.Normal(loc = self.mean, scale = self.std)
        return normal_dist.icdf(x)

    def pdf(self, x):
        normal_dist = dist.Normal(loc=self.mean, scale=self.std)
        return torch.exp(normal_dist.log_prob(x))
        
class StochDist():
    def __init__(self, *dist) -> None:
        self.dist = dist
        self.n_dist = len(dist)

    def sample(self, n_samples, device = "cpu", dtype = torch.float64):
        sample = torch.zeros((n_samples, self.n_dist), device= device, dtype = dtype)

        for idx, dist in enumerate(self.dist):
            sample[:,idx] = dist.sample(n_samples, device, dtype)
        
        return sample
    
    def eval_jointpdf(self, xi):
        fz = torch.ones(xi.shape[0])
        for i, xi_i in enumerate(xi.t()):
            fz = fz * self.dist[i].pdf(xi_i)
            
        return fz

class DesignDist():
    "This class represents a centered distribution around a design point"
    def __init__(self, *dist) -> None:
        self.dist = dist
        self.n_dist = len(dist)

    def sample(self, x_nom, n_samples, device = "cpu", dtype = torch.float64):
        centered_sample = torch.zeros((n_samples, self.n_dist), device= device, dtype = dtype)

        for idx, dist in enumerate(self.dist):
            centered_sample[:,idx] = dist.sample(n_samples, device, dtype)
        
        sample = centered_sample + x_nom
        return sample

    def eval_jointpdf(self, x_nom, x, mask):
        u = x - x_nom
        fz = torch.ones(x.shape[0])
        stoch_dist = self.dist[mask]
        for i, u_i in enumerate(u):
            fz = fz * stoch_dist[i].pdf(u_i)

        return fz

############################
# Gaussiann process helper #
############################

def get_matherons_path(gp_model, n_samp_gp, device = "cpu"):
    model_copy = copy.deepcopy(gp_model)
    model_copy = model_copy.float().to(device)
    path = draw_matheron_paths(model=model_copy, sample_shape=torch.Size([n_samp_gp]))
    return path

def get_gp_model(ro_data, x_map, mean = "const", add_noise = 1e-5, max_attempts = 5, logger = None):
    output_std = Standardize(m = 1)
    if mean == "const": mean_mod = ConstantMean()
    x, y = ro_data.get_all_samples()

    cov = ScaleKernel(
        base_kernel=MaternKernel(
            nu=2.5,
            ard_num_dims=x.shape[-1],
            lengthscale_prior=GammaPrior(3.0, 6.0),
        ),
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
    
    add_noise_untransformed = add_noise*torch.var(y)

    gp_model = SingleTaskGP(x,
                        y, 
                        mean_module= mean_mod,
                        covar_module= cov,
                        input_transform= x_map,
                        outcome_transform= output_std,
                        train_Yvar=torch.full_like(y, add_noise_untransformed))
    mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)

    start = time.time()    
    fit_gpytorch_mll(mll, pick_best_of_all_attempts = True, max_attempts = max_attempts)

    # print("Learned hyperparameters:")
    # logger.debug(f"Mean constant:{gp_model.mean_module.constant.item()}")
    # logger.debug(f"Lengthscale:{gp_model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()}")
    # logger.debug(f"Outputscale:{gp_model.covar_module.outputscale.item()}")

    if logger is not None: logger.debug(f"     Timing: Hyperpara tuning t = {time.time() - start:.3f}")
    return gp_model

def copy_gp_object(gp_model, device = "cpu", dtype = torch.float64):
    model_copy = copy.deepcopy(gp_model)
    model_copy = model_copy.to(dtype).to(device)
    return model_copy
