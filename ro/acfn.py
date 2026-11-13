"""Acquisition-function optimization.

This module provides functionality for optimizing acquisition functions
within the MLRO approach. It includes methods for estimating robustness
measures and their associated uncertainty.

The following acquisition-function classes are provided:

- ``AcfnUQ``: Acquisition function for uncertainty quantification.
- ``AcfnJEI``: Acquisition function for robust optimization based on an
  expected-improvement criterion.
- ``AcfnEM``: Acquisition function for robust optimization based on global
  mean optimization.
- ``AcfnTR``: Acquisition function for robust optimization using a local
  trust-region algorithm.

Author
------
Julius Schultz

Date
----
2026-10-16

"""


import torch, time, copy
import torch.distributions as dist
import math
from torch.distributions import Normal
from ro.helper import get_matherons_path

class RobustnessMeasure:
    """Class to compute robustness measure via a GP model.
       The class provides methods to push forward the GP model uncertainty to the robustness measure J, resulting in a sample of J.
    """
    def __init__(self, measure, sampler, J_device, model, ref_stat, norm_mean_std = None, tau = None, alpha = None, ref_q = 0.5, verbose = False, logger = None):
        self.measure = measure
        self.tau = tau
        self.alpha = alpha
        self.norm_mean_std = norm_mean_std
        self.sampler = sampler
        self.J_device = J_device
        self.model = model
        self.verbose = verbose
        self.ref_stat = ref_stat
        self.ref_q = ref_q
        self.logger = logger

    def get_norm_mean_std(self):
        normalization = self.model.get_mean_std_normalization()
        self.norm_mean_std = normalization

    def get_J_sample(self, x, gp_model, n_samp_J = 1000, n_samp_gp = 1000, matheron_path = None):
        """Computes J for each realization of a GP. Two options: classical sampling and sampling via matherons rule are available.
        """
        if self.sampler == "matheron":
            if matheron_path is None:
                if self.logger is not None: t0 = time.time()
                matheron_path = get_matherons_path(gp_model, n_samp_gp, device = self.J_device)
                if self.logger is not None: self.logger.debug(f"     Comp matheron path = {time.time() - t0:.3f} s")
            J_sample = self.get_J_sample_matheron(x, matheron_path, n_samp_J= n_samp_J)

        elif self.sampler == "classical":
            if self.J_device == "cuda":
                gp_model = copy.deepcopy(gp_model)
                gp_model.to(self.J_device)
            J_sample = self.get_J_sample_classical(x, gp_model, n_samp_J = n_samp_J, n_samp_gp = n_samp_gp)

        return J_sample.to("cpu")
    
    def get_J_sample_classical(self, x, gp_model, n_samp_J = 1000, n_samp_gp = 1000):
        """Computes J for each realization of a GP via classical sampling.
        """
        dtype = gp_model.train_inputs[0].dtype
        device = gp_model.train_inputs[0].device
        xi = self.model.xi_dist.sample(n_samp_J, device = device, dtype = dtype)
        x = self.model.x_dist.sample(x_nom = x, n_samples = n_samp_J)
        x = x.to(dtype).to(device)
        xi_at_x = torch.concatenate((x,xi), axis = 1)

        if self.verbose: start = time.time()
        with torch.no_grad():
            post = gp_model.posterior(xi_at_x)
            samples = post.sample(torch.Size([n_samp_gp]))[:,:,0]
        if self.verbose: self.logger.log(0, f"Sampling = {time.time() - start}")

        if self.verbose: start = time.time()
        if self.measure == "quantile":
            J_sample = torch.quantile(samples, self.tau, dim = 1)
        elif self.measure == "mean_std":
            J_sample = (1-self.alpha) / self.norm_mean_std[0] * torch.mean(samples, dim = 1) + self.alpha / self.norm_mean_std[1] * torch.std(samples, dim = 1)
        if self.verbose: self.logger.log(0, f"J comp = {time.time() - start}")
        
        return J_sample

    def get_J_sample_matheron(self, x, path, n_samp_J = 1000):
        """Computes J for each realization of a GP via matherons rule.
        """
        # print("Sampled with Matheron - start")
        if self.verbose: start = time.time()
        xi = self.model.xi_dist.sample(n_samp_J, device = self.J_device, dtype = torch.float32)
        x = self.model.x_dist.sample(x_nom = x, n_samples = n_samp_J)
        x = x.to(torch.float32).to(self.J_device)
        xi_at_x = torch.concatenate((x,xi), axis = 1)

        if self.verbose: self.logger.log(0, f"xi sample = {time.time() - start}")

        if self.verbose: start = time.time()
        samples = path(xi_at_x).detach()
        if self.verbose: self.logger.log(0, f"Path sample = {time.time() - start}")

        if self.measure == "quantile":
            J_sample = torch.quantile(samples, self.tau, dim = 1)
        elif self.measure == "mean_std":
            J_sample = (1-self.alpha) / self.norm_mean_std[0] * torch.mean(samples, dim = 1) + self.alpha / self.norm_mean_std[1] * torch.std(samples, dim = 1)
        if self.verbose: self.logger.log(0, f"J comp = {time.time() - start}")
                
        return J_sample

    def get_gp_mean_J(self, x, gp_model, n_samp_J = 1000, n_rep = 1):
        """Computes J based on the mean of the GP. The computation can be repreated for different input samples.
        """
        dtype = gp_model.train_inputs[0].dtype
        device = gp_model.train_inputs[0].device

        J_list = []
        x = self.model.x_dist.sample(x_nom = x, n_samples = n_samp_J)
        x = x.to(dtype).to(device)

        for i in range(n_rep):
            xi = self.model.xi_dist.sample(n_samp_J, device = device, dtype = dtype)
            xi_at_x = torch.concatenate((x,xi), axis = 1)

            if self.logger is not None: t0 = time.time()
            with torch.no_grad():
                post = gp_model.posterior(xi_at_x)
                mean = post.mean
            if self.measure == "quantile":
                J_list.append(torch.quantile(mean, self.tau, dim = 0)[0])
            elif self.measure == "mean_std":
                J_list.append((1-self.alpha) / self.norm_mean_std[0] * torch.mean(mean, dim = 0) + self.alpha / self.norm_mean_std[1] * torch.std(mean, dim = 0))
            
        J = torch.mean(torch.tensor(J_list))
        if self.logger is not None: self.logger.log(0, f"     Comp gp mean J = {time.time() - t0:.3f} s")
        return J.to("cpu")

    def get_min_J(self, gp_model, model, x_points, n_samp_gp, n_samp_J):
        """Iterates through a list of design points and gets the point with the minimum J (mean, std, ref_stat and x).
        """

        # Get x_points that fullfill constraints
        const_fullfilled = model.eval_constraint_on_sample(x_points)
        x_points = x_points[const_fullfilled]

        J_mean_vec = torch.zeros(x_points.shape[0])
        J_std_vec = torch.zeros(x_points.shape[0])
        J_ref_vec = torch.zeros(x_points.shape[0])

        if self.sampler == "matheron":
            path = get_matherons_path(gp_model, n_samp_gp, self.J_device)
        else:
            path = None

        for idx, x_c in enumerate(x_points):
            J_mean_vec[idx], J_std_vec[idx], J_ref_vec[idx] = self.get_J_sample_moments(x=x_c, gp_model = gp_model, n_samp_J=n_samp_J, n_samp_gp=n_samp_gp, matheron_path=path, include_ref=True)

        idx_min_ref = torch.argmin(J_ref_vec)
        min_J = J_mean_vec[idx_min_ref]
        min_J_std = J_std_vec[idx_min_ref]
        min_J_ref = J_ref_vec[idx_min_ref]
        x_star = x_points[idx_min_ref,:model.n_x]

        return x_star, min_J, min_J_std, min_J_ref

    def get_J_sample_ref(self, x, gp_model, n_samp_J = 1000, n_samp_gp = 1000, matheron_path = None):
        sample = self.get_J_sample(x, gp_model, n_samp_J, n_samp_gp, matheron_path)
        ref_J = self.get_ref_stat(sample)
        return ref_J
    
    def get_ref_stat(self, sample):
        """Gets the reference value of a J sample.
        """
        if self.ref_stat == "mean":
            ref = torch.mean(sample)
        elif self.ref_stat == "quant":
            ref = torch.quantile(sample, q = self.ref_q)  
        return ref
    
    def get_J_sample_moments(self, x, gp_model, n_samp_J = 1000, n_samp_gp = 1000, matheron_path = None, include_ref = False):
        """Gets the mean and std of a J sample
        """
        sample = self.get_J_sample(x, gp_model, n_samp_J, n_samp_gp, matheron_path)
        mean_J = torch.mean(sample)
        std_J = torch.std(sample)
        if include_ref: 
            ref_J = self.get_ref_stat(sample)
            return mean_J, std_J, ref_J
        else:
            return mean_J, std_J

class AcfnUQ:
    """This class contains aquisition functions for the stochastic domain.
    """
    def __init__(self, gp_model, model, x, J, measure):
        self.gp_model = gp_model
        self.model = model
        self.J = J
        self.x = x
        self.measure = measure

    def obj_fun_UFun_np(self, xi):
        xi = torch.from_numpy(xi.T)
        ufun = self.get_ufun(xi).cpu().numpy()
        return ufun

    def obj_fun_SE_np(self, z):
        z = torch.from_numpy(z.T)
        se = self.get_se(z).cpu().numpy()
        if self.measure == "mean_std": 
            x, xi = self.model.get_x_xi_from_z(self.x, z)
            f_xi = self.model.xi_dist.eval_jointpdf(xi)
            if torch.sum(self.model.stoch_design_mask) > 0:
                f_x = self.model.x_dist.eval_jointpdf(self.x, x, self.model.stoch_design_mask)
            else: f_x = 1
            f_z = f_xi * f_x
        else: f_z = torch.tensor(1.0)
        return -se*f_z.numpy()
    
    def get_ufun(self, xi, pdf_weight = False):
        n_points = xi.shape[0]
        xtilde = torch.zeros((n_points, self.model.n_dim))           
        x_det = self.x[self.model.det_design_mask]
        x  = x_det.repeat(n_points,1)
        xtilde[:,self.model.det_mask] = x
        xtilde[:,self.model.stoch_mask] = xi


        with torch.no_grad():
            post = self.gp_model.posterior(xtilde)
            mean = post.mean
            lower, upper = post.confidence_region()
            std = (upper - lower)/2
            std = std.unsqueeze(-1)
        if pdf_weight:
            f_y = self.model.xi_dist.eval_jointpdf(xi)
            ufun = torch.abs(self.J - mean)/(std * f_y)
        else:
            ufun = torch.abs(self.J - mean)/(std)
        return ufun[:,0]
    
    def get_se(self, xi):
        n_points = xi.shape[0]
        xtilde = torch.zeros((n_points, self.model.n_dim))           
        x_det = self.x[self.model.det_design_mask]
        x  = x_det.repeat(n_points,1)
        xtilde[:,self.model.det_mask] = x
        xtilde[:,self.model.stoch_mask] = xi

        with torch.no_grad():
            post = self.gp_model.posterior(xtilde)
            mean = post.mean
            lower, upper = post.confidence_region()
            std = (upper - lower)/2
        return std

class AcfnJEI:
    def __init__(self, gp_model, J_min, J_sampler: RobustnessMeasure, args, i_EI) -> None:
        self.gp_model = gp_model
        self.J_sampler = J_sampler
        self.J_min = J_min
        self.obj_fun_calls = 0
        self.n_explor = args.n_EX
        self.i_EI = i_EI
        self.EI_type = args.EI_type
        self.W2_dist = []
        if self.J_sampler.sampler == "matheron":
            self.path = get_matherons_path(self.gp_model, args.n_samp_gp_opt, self.J_sampler.J_device)
        else:
            self.path = None

    def obj_fun(self, x, n_samp_J, n_samp_gp, verbose = False):

        x = torch.from_numpy(x)
        sample = self.J_sampler.get_J_sample(x, self.gp_model, n_samp_J, n_samp_gp, self.path)
        J_mean = torch.mean(sample)
        J_std = torch.std(sample)

        # compute W2 distance
        # u = (torch.arange(1, len(sample)+1) - 0.5) / len(sample)
        # z_sorted, _ = torch.sort(sample)
        # normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        # z_theory = quant_mean + quant_std* normal.icdf(u) 
        # W2_dist = torch.sqrt(torch.mean((z_sorted - z_theory)**2)) # / quant_std
        # self.W2_dist.append(W2_dist)
        # print(f"quantile-RMSE to N(0,1) = {w2_dist:.5f}")

        if False:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.hist(sample, bins= 25)
            ax.set_title(f"W2 dist = {W2_dist}")
            plt.show()

        if self.i_EI < self.n_explor:
            JEI = self.J_EX(J_mean, J_std)
        else:
            if self.EI_type == "analytical":
                JEI = self.J_EI(J_mean, J_std)
            if self.EI_type == "logEI":
                JEI = self.J_logEI(J_mean, J_std)
            elif self.EI_type == "sampling":
                JEI = self.J_EI_sampling(sample)

        self.obj_fun_calls += 1
        if verbose:
            print(f"x = {x}, EI = {-JEI}")

        return -JEI.cpu().numpy()

    def J_EI(self, J_mean, J_std):
        if J_std < 1e-6:
            J_std = 1e-6
        standard_normal = dist.Normal(0, 1)
        JEI = (self.J_min - J_mean) * standard_normal.cdf((self.J_min - J_mean)/J_std) + J_std * torch.exp(standard_normal.log_prob((self.J_min - J_mean)/J_std))
        return JEI

    def log1mexp(self, x):
        log2 = math.log(2.0)
        return torch.where(
            x < -log2,
            torch.log1p(-torch.exp(x)),
            torch.log(-torch.expm1(x))
    )

    def J_logEI(self, J_mean, J_std, eps=1e-12):
        J_std = torch.clamp(J_std, min=eps)

        normal = Normal(
            torch.tensor(0.0, device=J_mean.device, dtype=J_mean.dtype),
            torch.tensor(1.0, device=J_mean.device, dtype=J_mean.dtype),
        )

        z = (self.J_min - J_mean) / J_std
        log_pdf = normal.log_prob(z)
        log_cdf = torch.special.log_ndtr(z)
        log_sigma = torch.log(J_std)

        log_abs_term = torch.log(torch.abs(z).clamp_min(eps)) + log_cdf

        pos_part = torch.logsumexp(
            torch.stack([log_abs_term, log_pdf], dim=0),
            dim=0
        )

        diff = log_abs_term - log_pdf
        neg_part = log_pdf + self.log1mexp(diff)

        log_inner = torch.where(z >= 0, pos_part, neg_part)

        return log_sigma + log_inner

    def J_EI_sampling(self, sample):
        I = self.J_min - sample
        I[I<0] = 0
        JEI = torch.mean(I)
        return JEI
    
    def J_EX(self, J_mean, J_std):
        if J_std < 1e-6:
            J_std = 1e-6
        standard_normal = dist.Normal(0, 1)
        JEX = J_std * torch.exp(standard_normal.log_prob((self.J_min - J_mean)/J_std))
        return JEX

    def sweep_obj(self, x_vec, n_samp_J, n_samp_gp, n_rep = 3):
        path = get_matherons_path(self.gp_model, n_samp_gp, self.J_sampler.J_device)
        JEI = torch.zeros((x_vec.size, n_rep))
        JEI_samp = torch.zeros((x_vec.size, n_rep))
        LogJEI_samp = torch.zeros((x_vec.size, n_rep))
        J_mean_vec = torch.zeros(x_vec.size, n_rep)
        J_std_vec = torch.zeros(x_vec.size, n_rep)
        for i in range(n_rep):
            for idx,x in enumerate(x_vec.nodes):
                sample = self.J_sampler.get_J_sample(torch.from_numpy(x), self.gp_model, n_samp_J, n_samp_gp, path)
                J_mean_vec[idx, i], J_std_vec[idx, i] = torch.mean(sample), torch.std(sample)
                if self.i_EI < self.n_explor: JEI[idx, i] = self.J_EX(J_mean_vec[idx,i], J_std_vec[idx,i])
                else: 
                    JEI[idx, i] = self.J_EI(J_mean_vec[idx,i], J_std_vec[idx,i])
                    JEI_samp[idx, i] = self.J_EI_sampling(sample)
                    LogJEI_samp[idx, i] = self.J_logEI(J_mean_vec[idx,i], J_std_vec[idx,i])

        return JEI, JEI_samp, LogJEI_samp, J_mean_vec, J_std_vec
    
class AcfnEM:
    '''
    Optimizes the expected robustness measure.
    '''
    def __init__(self, gp_model, J_sampler: RobustnessMeasure, args) -> None:
        self.gp_model = gp_model
        self.J_sampler = J_sampler
        self.obj_fun_calls = 0
        if self.J_sampler.sampler == "matheron":
            self.path = get_matherons_path(self.gp_model, args.n_samp_gp_opt, self.J_sampler.J_device)
        else:
            self.path = None

    def obj_fun_min_J(self, x, n_samp_J, n_samp_gp):

        x = torch.from_numpy(x)
        J_mean = torch.mean(self.J_sampler.get_J_sample(x, self.gp_model, n_samp_J, n_samp_gp, self.path))
        self.obj_fun_calls += 1

        return J_mean.cpu().numpy()

class AcfnTR:
    """Optimizes the quantile in a bounded trust region. The quantile is computed based on the GP mean prediciton.
    """
    def __init__(self, gp_model, model, x_star, J_sampler: RobustnessMeasure, delta = None, delta_bounds = [0.1, 0.25]):
        self.J_sampler = J_sampler
        if self.J_sampler.J_device == "cuda":
            model_copy = copy.deepcopy(gp_model)
            model_copy = model_copy.float().to(J_sampler.J_device)
            self.gp_model = model_copy
        else:
            self.gp_model = gp_model
        self.model = model

        x_bounds = torch.tensor(model.x_bounds)
        domain_lb = x_bounds[:,0]
        domain_ub = x_bounds[:,1]
        domain_interval = domain_ub - domain_lb

        if delta is None:
            self.delta = domain_interval * delta_bounds[1]
        else:
            self.delta = delta
        self.trust_reg = torch.vstack((torch.max(x_star - self.delta, domain_lb), torch.min(x_star + self.delta, domain_ub)))

    def obj_fun_TR_numpy(self, x, n_samp_J, n_rep_J):
        x = torch.from_numpy(x)
        J = self.J_sampler.get_gp_mean_J(x = x, gp_model = self.gp_model, n_samp_J= n_samp_J, n_rep = n_rep_J)

        return J.cpu().numpy()
    
    def sweep_obj(self, x_domain, n_samp_J):
        J = torch.zeros(len(x_domain))

        for idx,x in enumerate(x_domain):
            J[idx] = self.J_sampler.get_gp_mean_J(x = x, gp_model = self.gp_model, n_samp_J= n_samp_J, n_rep = 1)  
  
        return J

def comp_TR_radius(x_star, J_star, x_c, J_c, J_c_predict, model, delta_bounds = [0.1, 0.25], independent_TR = True, c1 = 0.75, c2 = 1.25, r1 = 0.25, r2 = 0.75, logger = None):

        x_bounds = torch.tensor(model.x_bounds)
        domain_interval = x_bounds[:,1] - x_bounds[:,0]
        delta_min = domain_interval * delta_bounds[0]
        delta_max = domain_interval * delta_bounds[1]

        actual_improve = torch.max(J_star - J_c,torch.tensor(0.0))      # if J_c > J_star then we do not have any improvement 
        predicted_improve = J_star - J_c_predict

        r = actual_improve/predicted_improve

        if logger is not None: logger.info(f"    J*, J_pred, J_c: {J_star:.3f},{J_c_predict:.3f}, {J_c:.3f}")
        if logger is not None: logger.info(f"    Actual improve:   ai = {actual_improve}")
        if logger is not None: logger.info(f"    Predicted improve:   spi = {predicted_improve}")
        if logger is not None: logger.info(f"    AI / SPI = {r}")

        if r < r1:      # Surrogate performed badly
            if independent_TR:
                delta = torch.max(c1*torch.abs(x_c - x_star), delta_min)
            else:
                delta = torch.ones(x_c.shape[0])*torch.max(c1*torch.norm(x_c - x_star), torch.norm(delta_min))
            if logger is not None: logger.info(f"    Shrink TR: New delta = {delta}")
        elif r > r2:    # Surrogate performed well
            # delta = torch.min(torch.max(c2*torch.abs(x_c - x_star),delta_min), delta_max)
            if independent_TR:
                delta = torch.min(c2*torch.abs(x_c - x_star), delta_max)
            else: 
                delta = torch.ones(x_c.shape[0])*torch.min(c2*torch.norm(x_c - x_star), torch.norm(delta_max))
            if logger is not None: logger.info(f"    Increase TR: New delta = {delta}")
        else:
            if independent_TR:
                delta = torch.max(torch.abs(x_c - x_star), delta_min)
            else:
                delta = torch.ones(x_c.shape[0])*torch.max(torch.norm(x_c - x_star), torch.norm(delta_min))
            if logger is not None: logger.info(f"    Keep TR: New delta = {delta}")

        return delta