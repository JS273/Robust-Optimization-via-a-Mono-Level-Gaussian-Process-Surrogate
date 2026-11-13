"""Benchmark and application models for robust optimization.

This module defines a common model interface for robust optimization.
Each model specifies:

- Bounds for the design variables.
- Uncertainty distributions for stochastic design variables.
- Probability distributions for uncertain parameters.
- Optional design constraints.
- An ``eval`` method for evaluating the model at a given design vector
  ``x`` and uncertain parameter vector ``xi``.

The ``eval`` method may evaluate an analytical benchmark function or call
a computationally intensive model, such as a finite-element simulation.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

import torch
import numpy as np
from scipy.optimize import NonlinearConstraint 
from botorch.models.transforms.input import Normalize
from ro.helper import StochDist, DesignDist, NormalDist, UniformDist, BetaDist, DiracDist
from utils.datastructures import Grid

try:
    from models.bracket import sim_bracket_qoi, volume_bracket
    HAS_BRACKET_MODEL = True
except ImportError:
    sim_bracket_qoi = None
    volume_bracket = None
    HAS_BRACKET_MODEL = False

def get_model(name, **kwargs):
    if name == "Sabater1d":
        model = Sabater1d(**kwargs)
    if name == "Sabater1dMod":
        model = Sabater1dMod(**kwargs)
    elif name == "Sabater2d":
        model = Sabater2d(**kwargs)
    elif name == "Sabater2dAdvancedUQ":
        model = Sabater2dAdvancedUQ(**kwargs)
    elif name == "SabaterND":
        model = SabaterND(**kwargs)
    elif name == "BraninHooUQ":
        model = BraninHooUQ(**kwargs)
    elif name == "Hartmann6":
        model = Hartmann6(**kwargs)
    elif name == "Bracket":
        model = Bracket(**kwargs)
    return model

class Model:
    def __init__(self, n_x, x_bounds, xi_dist, name, x_dist = None,) -> None:
        self.n_x = n_x
        self.n_xi = xi_dist.n_dist
        self.n_dim = self.n_x + self.n_xi
        self.x_bounds = x_bounds
        self.x_dist = x_dist
        self.xi_dist = xi_dist
        self.name = name
        self.constraints = ()
        self.constraint_satisfaction = 0.99     # Probability of consrtaint statisfaction in case of random design variables
        default_des_dist = DiracDist()
        if x_dist is None: self.x_dist = DesignDist(*([default_des_dist] * self.n_x))
        else: self.x_dist = x_dist
        self.stoch_mask, self.det_mask, self.stoch_design_mask, self.det_design_mask = self.classify_vars(self.x_dist, self.xi_dist)
        self.n_z = torch.sum(self.stoch_mask).item()
        
        x_stoch_bounds = self.get_x_stoch_bounds() 
        self.x_bounds_incl_uq = (torch.tensor(self.x_bounds) + x_stoch_bounds).tolist()

        self.xi_bounds = [dist.get_bounds() for dist in self.xi_dist.dist]

        self.xtilde_map_det_design = Normalize(d = self.n_dim, 
                                    bounds = torch.tensor(self.x_bounds + self.xi_bounds).T)
        self.xtilde_map = Normalize(d = self.n_dim, 
                                    bounds = torch.tensor(self.x_bounds_incl_uq + self.xi_bounds).T)
        self.x_map = Normalize(d = self.n_x, 
                                bounds = torch.tensor(self.x_bounds_incl_uq).T)
        self.xi_map = Normalize(d = self.n_xi, 
                                bounds = torch.tensor(self.xi_bounds).T)

    def eval(self, x, Xi):
        pass

    def eval_xtilde(self, xtilde):
        xtilde = torch.atleast_2d(xtilde)
        y = self.eval(xtilde[:,:self.n_x], xtilde[:,self.n_x:])
        return y
    
    def eval_const(self, x):
        "Unconstraint as default behaviour. Thus all points in x are feasible and hence g(x) <= 0.0"
        c = torch.rand(size = (x.shape[0],)) - 2
        return c

    def eval_stoch_const(self, x):
        n_samp = int(1e6)
        x = self.x_dist.sample(x_nom = x, n_samples = n_samp)
        c_sample = self.eval_const(x)
        c = torch.quantile(c_sample, self.constraint_satisfaction)
        return c

    def eval_const_numpy(self, x):
        """ Wrapper function to evaluate the constraint function g(x) for numpy datatypes.
            If any design variable is subject to uncertainty then the 99% quantile of g(x) is considered.
            Thus, we ensure satisfaction of constraints by 99% probability 

        Args:
            x (numpy.array): input design vector

        Returns:
            c (float): constraint function evaluation.
        """
        x = torch.from_numpy(x)

        if torch.sum(self.stoch_design_mask).item() > 0:
            c = self.eval_stoch_const(x).numpy()
        else:
            x = torch.atleast_2d(x) 
            c = self.eval_const(x).numpy()
        return c
    
    def eval_constraint_on_sample(self, x_sample):
        """ Evaluates the constraint on a vector of design points 

        Args:
            x_sample (torch.tensor): input design points of shape (n x d)

        Returns:
            List (boolean): List with constraint evaluation
        """
        
        x_sample = torch.atleast_2d(x_sample)
        const_eval = []
        for x in x_sample:
            if torch.sum(self.stoch_design_mask).item() > 0:
                c = self.eval_stoch_const(x)
            else:
                x = torch.atleast_2d(x) 
                c = self.eval_const(x)
            const_eval.append((c <= 0.0).item())
        return const_eval

    def get_x_stoch_bounds(self):
        '''Gets the bounds of the uncertainty in the design variables'''
        bounds = torch.zeros((self.n_x, 2))
        for i, dist in enumerate(self.x_dist.dist):
            if type(dist) is not DiracDist:
                dist_bounds = dist.get_bounds()
                bounds[i] = torch.tensor(dist_bounds)

        return bounds

    def get_xgrid(self, n_points = 10):
        if self.n_x == 1: return Grid(np.linspace(self.x_bounds[0][0], self.x_bounds[0][1], n_points))
        elif self.n_x == 2: return Grid(np.linspace(self.x_bounds[0][0], self.x_bounds[0][1], n_points), np.linspace(self.x_bounds[1][0], self.x_bounds[1][1], n_points))

    def get_xigrid(self, n_points = 10):
        if self.n_x == 1: return Grid(np.linspace(self.xi_bounds[0][0], self.xi_bounds[0][1], n_points))
        elif self.n_x == 2: return Grid(np.linspace(self.xi_bounds[0][0], self.xi_bounds[0][1], n_points), np.linspace(self.xi_bounds[1][0], self.xi_bounds[1][1], n_points))

    def get_xtilde(self, x, z):
        xtilde = torch.zeros(self.n_dim)
        if torch.any(self.det_design_mask):
            x_deterministic = x[self.det_design_mask]
            xtilde[self.stoch_mask] = z
            xtilde[self.det_mask] = x_deterministic
        else:
            xtilde = z

        return xtilde

    def classify_vars(self, x_dist, xi_dist):
        det_design_var_mask = [type(x) is DiracDist for x in x_dist.dist]
        det_immutable_var_mask = [False]*xi_dist.n_dist

        det_mask = torch.tensor(det_design_var_mask + det_immutable_var_mask)
        stoch_mask = ~det_mask
        det_design_mask = torch.tensor(det_design_var_mask)
        stoch_design_mask = ~det_design_mask
        return stoch_mask, det_mask, stoch_design_mask, det_design_mask

    def get_stoch_box_domain(self, x):
        # Check if we have a stochastic design variable
        if torch.any(self.stoch_design_mask):
            stoch_design_dist = [x for x, m in zip(self.x_dist.dist, self.stoch_design_mask) if m]
            stoch_design_domain = torch.tensor([dist.get_bounds() for dist in stoch_design_dist])
            stoch_design_domain = stoch_design_domain + x[self.stoch_design_mask].unsqueeze(-1)
            immutable_domain = torch.tensor(self.xi_bounds)
            stoch_box_domain = torch.concatenate((stoch_design_domain, immutable_domain), axis = 0)
        else:
            stoch_box_domain = torch.tensor(self.xi_bounds)
        return stoch_box_domain

    def get_x_xi_from_z(self, x_nom, z):
        '''Gets a x, xi from z and a nominal design point x_nom'''
        n_stoch_design = torch.sum(self.stoch_design_mask)
        if n_stoch_design == 0:
            xi = z
            x = x_nom
        else:
            xi = z[:,-self.n_xi:]
            x = z[:,:n_stoch_design]
        return x, xi

    def get_mean_std_normalization(self):
        # Compute normalization constants
        sobol_engine = torch.quasirandom.SobolEngine(dimension=self.n_dim, scramble=False, seed = 0)
        sobol_engine.fast_forward(0)
        init_u = sobol_engine.draw(n=self.n_dim * 5)
        init_xtilde = self.xtilde_map_det_design.untransform(init_u)
        init_y = self.eval(init_xtilde[:,:self.n_x], init_xtilde[:,self.n_x:])
        norm_mean = torch.mean(init_y)
        norm_std = torch.std(init_y)
        return [norm_mean, norm_std]

    def to_str(self, tau):
        return str(tau).replace(".", "p")

    def load_opt(self, measure = "quantile", alpha = None, tau = None, data_file = None):
        if data_file is None:
            if measure == "quantile":
                data_file = f"_reference_comp/{self.name}_tau{self.to_str(tau)}.pt" 
            elif measure == "mean_std":
                data_file = f"_reference_comp/{self.name}_alpha{self.to_str(alpha)}.pt" 
            
        data = torch.load(data_file, weights_only=False)
        return data.x_star, data.q_star
    
    def load_oat_plot(self, measure = "quantile", alpha = None, tau = None, data_file = None):
        if data_file is None:
            if measure == "quantile":
                data_file = f"_reference_comp/{self.name}_tau{self.to_str(tau)}.pt" 
            elif measure == "mean_std":
                data_file = f"_reference_comp/{self.name}_alpha{self.to_str(alpha)}.pt" 

        data = torch.load(data_file, weights_only=False)
        return data.oat_plot
    
    def load_domain_plot(self, measure = "quantile", alpha = None, tau = None, data_file = None):
        if data_file is None:
            if measure == "quantile":
                data_file = f"_reference_comp/{self.name}_tau{self.to_str(tau)}.pt" 
            elif measure == "mean_std":
                data_file = f"_reference_comp/{self.name}_alpha{self.to_str(alpha)}.pt" 
                
        data = torch.load(data_file, weights_only=False)
        return data.domain_plot

    def __getstate__(self):
        state = self.__dict__.copy()
        state["constraints"] = None
        return state
    
class Demo1d(Model):
    def __init__(self) -> None:

        n_x = 1
        x_bounds = [[0.0, 1.0]]
        xi_bounds = [[0.0, 1.0]]
        beta_dist = BetaDist(lb = xi_bounds[0][0], ub=xi_bounds[0][1], alpha = 2.0, beta = 2.0)
        xi_dist = StochDist(beta_dist)

        name = "demo_1d"
        super().__init__(n_x = n_x,
                    x_bounds = x_bounds,
                    xi_dist = xi_dist,
                    name = name)
        
    def eval(self, x, xi):
        fx = 1.0
        fy = 1.5
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)
        y = (0.5 - 1.0* x[:,0] + torch.sin(2*torch.pi*(fx + 0.5 * xi[:,0])*x[:,0] -torch.pi/2) + 1.5* xi[:,0] + 0.5* torch.cos(2*torch.pi*(fy-0.5*x[:,0])*xi[:,0] + torch.pi) +3 )/5
        return y.unsqueeze(-1)
    
class Sabater1d(Model):
    def __init__(self, design_uq = True) -> None:
        if design_uq: name = "Sabater1d_DesignUQ"
        else: name = "Sabater1d"

        # Define immutable variables (i.e. random variables which are not design variables)
        xi1_bounds = [0.0, 10.0]
        uniform_dist = UniformDist(low = xi1_bounds[0], high = xi1_bounds[1])
        normal_dist = NormalDist(mean = 1.01, std = 0.71)
        xi_dist = StochDist(uniform_dist, normal_dist)

        # Define design variables
        n_x = 1
        x_bounds = [[2.0, 8.0]]
        if design_uq:
            x_d_dist = UniformDist(low = -0.2, high= 0.2)
            x_dist = DesignDist(x_d_dist)
        else: 
            x_dist = None

        self.metadata = {"name": name,
                         design_uq: design_uq}

        super().__init__(n_x = n_x,
                         x_bounds = x_bounds,
                         xi_dist = xi_dist,
                         x_dist = x_dist,
                         name = name)

    #     const = NonlinearConstraint(
    #         self.eval_const_numpy,
    #         -np.inf,
    #         0.0
    #     )

    #     self.constraints = (const,)

    # def eval_const(self, x):
    #     "Gets x in physical coordinates"
    #     "Returns the classical constraint formulation g(x) <= 0"
    #     c = 5 - x
    #     return c

    def eval(self, x, xi):
        a = 2.0
        b = 8.0
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)
        y = 3 - 4*torch.exp(-4*(x[:,0]-4)**2)-5.2*torch.exp(-4*(x[:,0]-6)**2) + (x[:,0]-a)/(b - a)*xi[:,0] + (b-x[:,0])/(b-a)*xi[:,1]
        return y.unsqueeze(-1)

class Sabater1dMod(Model):
    def __init__(self, design_uq = True) -> None:
        if design_uq: name = "Sabater1dMod_DesignUQ"
        else: name = "Sabater1dMod"

        # Define immutable variables (i.e. random variables which are not design variables)
        xi1_bounds = [0.0, 10.0]
        uniform_dist = UniformDist(low = xi1_bounds[0], high = xi1_bounds[1])
        normal_dist = NormalDist(mean = 1.01, std = 0.71)
        xi_dist = StochDist(uniform_dist, normal_dist)

        # Define design variables
        n_x = 1
        x_bounds = [[2.0, 8.0]]
        if design_uq:
            x_d_dist = UniformDist(low = -0.25, high= 0.25)
            x_dist = DesignDist(x_d_dist)
        else: 
            x_dist = None
            
        super().__init__(n_x = n_x,
                         x_bounds = x_bounds,
                         xi_dist = xi_dist,
                         x_dist = x_dist,
                         name = name)

    def eval(self, x, xi):
        a = 2.0
        b = 8.0
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)
        y = 3 - 4*torch.exp(-30*(x[:,0]-4)**2)-5.2*torch.exp(-2*(x[:,0]-6)**2) + (x[:,0]-a)/(b - a)*xi[:,0] + (b-x[:,0])/(b-a)*xi[:,1]
        return y.unsqueeze(-1)
    
class Sabater2d((Model)):
    def __init__(self, uq_domain = "linear") -> None:

        name = f"Sab2d_{uq_domain}_uq"

        # Define immutable variables (i.e. random variables which are not design variables)
        xi1_bounds = [0.0, 10.0]
        uniform_dist = UniformDist(low = xi1_bounds[0], high = xi1_bounds[1])
        normal_dist = NormalDist(mean = 1.01, std = 0.71)
        xi_dist = StochDist(uniform_dist, normal_dist)

        # Define design variables
        n_x = 2
        x_bounds = [[2.0, 8.0],[2.0, 8.0]]
        x_dist = None
        self.uq_domain = uq_domain

        self.metadata = {"name": name,
                         "uq_domain": uq_domain}

        super().__init__(n_x = n_x,
                        x_bounds = x_bounds,
                        xi_dist = xi_dist,
                        x_dist = x_dist,
                        name = name)

    def eval(self, x, xi):
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)
        a = 2.0
        b = 8.0
        if self.uq_domain == "linear":
            Y1 = 3 - 4*torch.exp(-4*(x[:,0]-4)**2)-5.2*torch.exp(-4*(x[:,0]-6)**2) + (x[:,0]-a)/(b - a)*xi[:,0] + (b-x[:,0])/(b-a)*xi[:,1]
            Y2 = 3 - 4*torch.exp(-4*(x[:,1]-4)**2)-5.2*torch.exp(-4*(x[:,1]-6)**2) + (x[:,1]-a)/(b - a)*xi[:,0] + (b-x[:,1])/(b-a)*xi[:,1]
        elif self.uq_domain == "quad":
            Y1 = 3 - 4*torch.exp(-4*(x[:,0]-4)**2)-5.2*torch.exp(-4*(x[:,0]-6)**2) + 0.1*(x[:,0]-a)/(b - a)*xi[:,0]**2 + 0.5*(b-x[:,0])/(b-a)*xi[:,1]**2
            Y2 = 3 - 4*torch.exp(-4*(x[:,1]-4)**2)-5.2*torch.exp(-4*(x[:,1]-6)**2) + 0.1*(x[:,1]-a)/(b - a)*xi[:,0]**2 + 0.5*(b-x[:,1])/(b-a)*xi[:,1]**2
        elif self.uq_domain == "nonlin":
            Y1 = 3 - 4*torch.exp(-4*(x[:,0]-4)**2)-5.2*torch.exp(-4*(x[:,0]-6)**2) + 0.1*(x[:,0]-a)/(b - a)*((xi[:,0]-5.0)/1.4)**4 + 0.3*(b-x[:,0])/(b-a)*xi[:,1]**2 + 5 * torch.sin(0.2*xi[:,0]*xi[:,1])**2
            Y2 = 3 - 4*torch.exp(-4*(x[:,1]-4)**2)-5.2*torch.exp(-4*(x[:,1]-6)**2) + 0.1*(x[:,1]-a)/(b - a)*((xi[:,0]-5.0)/1.4)**4 + 0.3*(b-x[:,1])/(b-a)*xi[:,1]**2 + 5 * torch.sin(0.2*xi[:,0]*xi[:,1])**2
        y = Y1 + Y2
        return y.unsqueeze(-1)
 
class BraninHooUQ((Model)):
    def __init__(self, uq_domain = "non-linear") -> None:

        name = f"BraninHooUQ_{uq_domain}_uq"

        # Define immutable variables (i.e. random variables which are not design variables)
        n_xi = 3
        xi_bounds = [[-np.pi, np.pi]]*n_xi
        uniform_dist = UniformDist(low = xi_bounds[0][0], high = xi_bounds[0][1])
        xi_dist = StochDist(uniform_dist, uniform_dist, uniform_dist)

        # Define design variables
        n_x = 2
        x_bounds = [[-5.0, 10.0],[0.0, 15.0]]
        x_dist = None

        self.uq_domain = uq_domain

        self.metadata = {"name": name,
                         "uq_domain": uq_domain}

        super().__init__(n_x = n_x,
                                x_bounds = x_bounds,
                                xi_dist = xi_dist,
                                x_dist = x_dist,
                                name = name)

    def eval(self, x, xi):
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)

        a = 1
        b = 5.1/(2*torch.pi)**2
        c = 5/torch.pi
        r = 6
        s = 10
        t = 1/(8*torch.pi)

        if self.uq_domain == "linear":
            Y_stoch = (x[:,0]-a)/(b - a)*xi[:,0] + (b-x[:,0])/(b-a)*xi[:,1] + (b-x[:,1])/(b-a) * xi[:,1] * xi[:,2]
        if self.uq_domain == "nonlin":
            Y_stoch = 5*(x[:,0] - self.x_bounds[0][0])  / (self.x_bounds[0][1]- self.x_bounds[0][0]) * torch.sin(xi[:,0]) +  5*(x[:,1] - self.x_bounds[1][0])  / (self.x_bounds[1][1]- self.x_bounds[1][0]) * xi[:,1]**2 +  0.1*(x[:,0] - 4)**2 * xi[:,2]
        elif self.uq_domain == "ishigami":
            Y_stoch = 10*(torch.sin(xi[:,0]) +  0.04*(x[:,0] - 4.0)**2 * torch.sin(xi[:,1])**2 + 0.5*(x[:,1] - self.x_bounds[1][0])  / (self.x_bounds[1][1]- self.x_bounds[1][0]) * xi[:,2]**4 * torch.sin(xi[:,0]))
        elif self.uq_domain == "test":
            Y_stoch = + 0.1*(x[:,0]-a)/(b - a)*((xi[:,0]-5.0)/1.4)**4 + 0.3*(b-x[:,0])/(b-a)*xi[:,1]**2 + 5 * torch.sin(0.2*xi[:,0]*xi[:,2])**2 + + 0.1*(x[:,1]-a)/(b - a)*((xi[:,0]-5.0)/1.4)**4 + 0.3*(b-x[:,1])/(b-a)*xi[:,1]**2 + 5 * torch.sin(0.2*xi[:,0]*xi[:,1])**2
        Y_design = a*(x[:,1] - b*x[:,0]**2 + c*x[:,0] - r)**2 + s * (1-t)*torch.cos(x[:,0]) + s + 2*x[:,0]

        Y = Y_stoch + Y_design

        return Y.unsqueeze(-1)
    
class SpatialNonLinear((Model)):
    def __init__(self) -> None:
        super().__init__(n_x = 2,
                         x_bounds = [[-5.0, 10.0],[0.0, 15.0]], 
                         name = f"BraninHooIshigami")
        self.n_xi = 3
        self.n_dim = self.n_xi + self.n_x
        
        self.xi_bounds = [[-np.pi, np.pi]]*self.n_xi

        self.xtilde_map = Normalize(d = self.n_dim, 
                                    bounds = torch.tensor(self.x_bounds + self.xi_bounds).T)
        self.x_map = Normalize(d = self.n_x, 
                                bounds = torch.tensor(self.x_bounds).T)
        self.xi_map = Normalize(d = self.n_xi, 
                                bounds = torch.tensor(self.xi_bounds).T)

        uniform_dist = UniformDist(low = self.xi_bounds[0][0], high = self.xi_bounds[0][1])
        self.xi_dist = StochDist(uniform_dist, uniform_dist, uniform_dist)


    def eval(self, x, xi):
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)

        ishi_a1 = 7.0
        ishi_a2 = 5.0
        ishi_b1 = 2.0
        ishi_b2 = 0.05

        # Y = torch.sin(xi[:,0]) + ishi_a1 + (x[:,1] - self.x_bounds[1][0])  * (ishi_a2-ishi_a1)/(self.x_bounds[1][1]- self.x_bounds[1][0]) * torch.sin(xi[:,1])**2 + ishi_b1 + (x[:,0] - self.x_bounds[0][0])  * (ishi_b2-ishi_b1)/(self.x_bounds[0][1]- self.x_bounds[0][0]) * xi[:,2]**4 * torch.sin(xi[:,0])
        # Y = torch.sin(xi[:,0]) + ishi_a1  * torch.sin(xi[:,1])**2 + ishi_b1 * xi[:,2]**4 * torch.sin(xi[:,0])
        # Y = torch.sin(xi[:,0]) + ishi_a1  * torch.sin(xi[:,1])**2 + ishi_b1 * xi[:,2]**2

        a = (ishi_a1 + (x[:,1] - self.x_bounds[1][0])  * (ishi_a2-ishi_a1)/(self.x_bounds[1][1]- self.x_bounds[1][0]))
        b = (ishi_b1 + (x[:,0] - self.x_bounds[0][0])  * (ishi_b2-ishi_b1)/(self.x_bounds[0][1]- self.x_bounds[0][0]))

        a = (x[:,1] - self.x_bounds[1][0])  / (self.x_bounds[1][1]- self.x_bounds[1][0])
        b = (self.x_bounds[0][1] - x[:,0])  / (self.x_bounds[0][1]- self.x_bounds[0][0])


        Y = torch.sin(xi[:,0]) + 2*a * torch.sin(xi[:,1])**2 + 0.1* b * xi[:,2]**2
            
        return Y.unsqueeze(-1)
    
class SpatialIshigami((Model)):
    def __init__(self) -> None:
        super().__init__(n_x = 2,
                         x_bounds = [[-5.0, 10.0],[0.0, 15.0]], 
                         name = f"BraninHooIshigami")
        self.n_xi = 3
        self.n_dim = self.n_xi + self.n_x
        
        self.xi_bounds = [[-np.pi, np.pi]]*self.n_xi

        self.xtilde_map = Normalize(d = self.n_dim, 
                                    bounds = torch.tensor(self.x_bounds + self.xi_bounds).T)
        self.x_map = Normalize(d = self.n_x, 
                                bounds = torch.tensor(self.x_bounds).T)
        self.xi_map = Normalize(d = self.n_xi, 
                                bounds = torch.tensor(self.xi_bounds).T)

        uniform_dist = UniformDist(low = self.xi_bounds[0][0], high = self.xi_bounds[0][1])
        self.xi_dist = StochDist(uniform_dist, uniform_dist, uniform_dist)


    def eval(self, x, xi):
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)

        ishi_a1 = 7.0
        # ishi_a2 = 5.0
        ishi_b1 = 0.1
        # ishi_b2 = 0.05

        # Y_stoch = torch.sin(xi[:,0]) + ishi_a1 * torch.sin(xi[:,1])**2 + ishi_b1 * xi[:,2]**4 * torch.sin(xi[:,0])
        Y_stoch = 10*(torch.sin(xi[:,0]) +  0.04*(x[:,0] - 4.0)**2 * torch.sin(xi[:,1])**2 + 0.5*(x[:,1] - self.x_bounds[1][0])  / (self.x_bounds[1][1]- self.x_bounds[1][0]) * xi[:,2]**4 * torch.sin(xi[:,0]))
        # Y_stoch = 25 * torch.sin(xi[:,0]) +  15*(self.x_bounds[1][0] - x[:,1])  / (self.x_bounds[1][1]- self.x_bounds[1][0]) * torch.sin(xi[:,1])**2 +  20*(self.x_bounds[0][1] - x[:,0])  / (self.x_bounds[0][1]- self.x_bounds[0][0]) * xi[:,2]**4 * torch.sin(xi[:,0])
        # Y = 10 * torch.sin(xi[:,0]) +  14*(self.x_bounds[1][0] - x[:,1])  / (self.x_bounds[1][1]- self.x_bounds[1][0]) * torch.sin(xi[:,1])**2 +  4*(self.x_bounds[0][1] - x[:,0])  / (self.x_bounds[0][1]- self.x_bounds[0][0]) * xi[:,2]**4 * torch.sin(xi[:,0])
        return Y_stoch.unsqueeze(-1)
    
class Sabater2dAdvancedUQ((Model)):
    def __init__(self, uq_domain = "default") -> None:
        super().__init__(x_bounds = [[2.0, 8.0],[2.0, 8.0]])
        self.n_dim = 5
        self.n_x = 2
        self.n_xi = 3
        
        self.xi_bounds = [[-np.pi, np.pi]]*self.n_xi
        self.xi_center = (self.xi_bounds[1]-self.xi_bounds[0])/2


        uniform_dist = UniformDist(low = self.xi_bounds[0][0], high = self.xi_bounds[0][1])

        self.xtilde_map = Normalize(d = self.n_dim, 
                                    bounds = torch.tensor(self.x_bounds + self.xi_bounds).T)
        self.x_map = Normalize(d = self.n_x, 
                                bounds = torch.tensor(self.x_bounds).T)
        self.xi_map = Normalize(d = self.n_xi, 
                                bounds = torch.tensor(self.xi_bounds).T)

        self.xi_dist = StochDist(uniform_dist, uniform_dist, uniform_dist)

        self.uq_domain = uq_domain

    def eval(self, x, xi):
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)
        a1 = 7.0
        a2 = 5.0
        b = 0.1
        if self.uq_domain == "default":
            Y1 = 3 - 4*torch.exp(-4*(x[:,0]-4)**2)-5.2*torch.exp(-4*(x[:,0]-6)**2) + torch.sin(xi[:,0]) + a1 - (x[:,0] - self.x_bounds[0][0])  * (a2-a1)/(self.x_bounds[0][1]- self.x_bounds[0][0])  * torch.sin(xi[:,1])**2 + b * xi[:,2]**4 * torch.sin(xi[:,0])
            Y2 = 3 - 4*torch.exp(-4*(x[:,1]-4)**2)-5.2*torch.exp(-4*(x[:,1]-6)**2) + torch.sin(xi[:,0]) + a1 - (x[:,1] - self.x_bounds[0][0])  * (a2-a1)/(self.x_bounds[0][1]- self.x_bounds[0][0]) * torch.sin(xi[:,1])**2 + b * xi[:,2]**4 * torch.sin(xi[:,0])
        y = Y1 + Y2
        return y.unsqueeze(-1)
    
class SabaterND(Model):
    def __init__(self, uq_domain = "linear", n_x = 3) -> None:
        super().__init__(n_x = n_x,
                         x_bounds = [[2.0, 8.0]]*n_x,
                         name = f"Sabater{n_x}d_{uq_domain}_uq")
        self.n_xi = 2
        self.n_dim = 2 + n_x

        xi1_bounds = [0.0, 10.0]
        uniform_dist = UniformDist(low = xi1_bounds[0], high = xi1_bounds[1])

        normal_dist = NormalDist(mean = 1.01, std = 0.71)
        xi2_lower = normal_dist.icdf(torch.tensor(0.01))
        xi2_upper = normal_dist.icdf(torch.tensor(0.99))
        xi2_bounds = [xi2_lower.item(), xi2_upper.item()]

        self.xi_bounds = [xi1_bounds, xi2_bounds]

        self.xtilde_map = Normalize(d = self.n_dim, 
                                    bounds = torch.tensor(self.x_bounds + self.xi_bounds).T)
        self.x_map = Normalize(d = self.n_x, 
                                bounds = torch.tensor(self.x_bounds).T)
        self.xi_map = Normalize(d = self.n_xi, 
                                bounds = torch.tensor(self.xi_bounds).T)

        self.xi_dist = StochDist(uniform_dist, normal_dist)

        self.uq_domain = uq_domain

    def eval(self, x, xi):
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)
        a = 2.0
        b = 8.0
        Y = 0
        if self.uq_domain == "linear":
            for i in range(x.shape[1]):
                Y += 3 - 4*torch.exp(-4*(x[:,i]-4)**2)-5.2*torch.exp(-4*(x[:,i]-6)**2) + (x[:,i]-a)/(b - a)*xi[:,0] + (b-x[:,i])/(b-a)*xi[:,1]
        elif self.uq_domain == "quad":
            for i in range(x.shape[1]):
                Y += 3 - 4*torch.exp(-4*(x[:,i]-4)**2)-5.2*torch.exp(-4*(x[:,i]-6)**2) + 0.1*(x[:,i]-a)/(b - a)*xi[:,0]**2 + (b-x[:,i])/(b-a)*xi[:,1]**2
        return Y.unsqueeze(-1)
    
class Hartmann6(Model):
    def __init__(self, n_x = 6, uq_domain = "uq") -> None:

        name = f"Hartmann6_{uq_domain}"

        # Immutable variables
        n_xi = 3
        xi_bounds = [[-np.pi, np.pi]]*n_xi
        uniform_dist = UniformDist(low = xi_bounds[0][0], high = xi_bounds[0][1])
        xi_dist = StochDist(uniform_dist, uniform_dist, uniform_dist)

        # Design variables
        n_x = 6
        x_bounds = [[0.0, 1.0]]*n_x
        x_dist = None

        self.uq_domain = uq_domain

        self.metadata = {"name": name,
                         "uq_domain": uq_domain}

        super().__init__(n_x = n_x,
                x_bounds = x_bounds,
                xi_dist = xi_dist,
                x_dist = x_dist,
                name = name)
        
    def eval(self, x, xi):
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)
        alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])  # (4,)
        A = torch.tensor([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])  # (4, 6)
        P = 1e-4 * torch.tensor([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])  # (4, 6)

        # xx shape: (n, 6) -> (n, 1, 6)
        # P shape: (4, 6) -> (1, 4, 6)
        diffs = x.unsqueeze(1) - P.unsqueeze(0)  # (n, 4, 6)

        inner = torch.sum(A * diffs ** 2, dim=2)  # (n, 4)

        terms = alpha * torch.exp(-inner)  # (n, 4)
        outer = torch.sum(terms, dim=1)    # (n,)


        #y = -(2.58 + outer) / 1.94 # 
        if self.uq_domain == "uq": y = -outer + 5*(x[:,0] - self.x_bounds[0][0])  / (self.x_bounds[0][1]- self.x_bounds[0][0]) * torch.sin(xi[:,0]) +  (x[:,1] - 0.5)**2 * xi[:,1]**2 +  0.5*(x[:,0] - 0.8)**2 * xi[:,2]
        elif self.uq_domain == "none": y = -outer
        return y.unsqueeze(-1)
    
class EggCrate(Model):
    def __init__(self) -> None:
        super().__init__(n_x = 2,
                         x_bounds = [[-5.0, 5.0]]*2,
                         name = f"EggCrate")
        self.n_xi = 2
        self.n_dim = 2 + self.n_x

        xi1_bounds = [0.0, 10.0]
        self.xi1_center = (xi1_bounds[1]-xi1_bounds[0])/2
        uniform_dist = UniformDist(low = xi1_bounds[0], high = xi1_bounds[1])

        normal_dist = NormalDist(mean = 1.01, std = 0.71)
        xi2_lower = normal_dist.icdf(torch.tensor(0.01))
        xi2_upper = normal_dist.icdf(torch.tensor(0.99))
        xi2_bounds = [xi2_lower, xi2_upper]

        self.xi_bounds = [xi2_lower.item(), xi2_upper.item()]

        self.xtilde_map = Normalize(d = self.n_dim, 
                                    bounds = torch.tensor(self.x_bounds + self.xi_bounds).T)
        self.x_map = Normalize(d = self.n_x, 
                                bounds = torch.tensor(self.x_bounds).T)
        self.xi_map = Normalize(d = self.n_xi, 
                                bounds = torch.tensor(self.xi_bounds).T)

        self.xi_dist = StochDist(uniform_dist, normal_dist)

    def eval(self, x, xi):
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)
        y_design = 25*(x[:,0]**2/25 + torch.sin(x[:,0])**2 + x[:,1]**2/25 + torch.sin(x[:,1])**2)
    
        a1 = self.x_bounds[0][0]
        b1 = self.x_bounds[0][1]
        a2 = self.x_bounds[1][0]
        b2 = self.x_bounds[1][1]
        y_stoch = 0.1*(x[:,0]-a1)/(b1 - a1)*xi[:,0]**2 + (b1-x[:,0])/(b1-a1)*xi[:,1]**2 + 0.1*(x[:,1]-a2)/(b2 - a2)*xi[:,0]**2 + (b2-x[:,1])/(b2-a2)*xi[:,1]**2
        y = y_design + y_stoch

        return y.unsqueeze(-1)

class Bracket(Model):
    def __init__(self, design_uq = True) -> None:
        if design_uq: name = "Bracket_DesignUQ"
        else: name = "Bracket"

        # Define immutable variables (i.e. random variables which are not design variables)
        normal_dist_1 = NormalDist(mean = 5000, std = 1000)
        normal_dist_2 = NormalDist(mean = 0.0, std = torch.pi/6)
        xi_dist = StochDist(normal_dist_1, normal_dist_2)

        # Define design variables
        n_x = 5
        x_bounds = [[60, 100], [10, 50], [10, 50], [60, 100], [10, 30]]

        if design_uq:
            x_d_dist = NormalDist(mean = 0.0, std = 0.1)
            x_d_dists = [x_d_dist]*n_x
            x_dist = DesignDist(*x_d_dists)
        else: 
            x_dist = None

        self.metadata = {"name": name,
                         design_uq: design_uq}

        super().__init__(n_x = n_x,
                         x_bounds = x_bounds,
                         xi_dist = xi_dist,
                         x_dist = x_dist,
                         name = name)

        # other deterministic immutable variables
        p1 = 50 # right outer edge
        p2 = 250 # bottom outer edge
        p3 = 250 # left outer edge
        p4 = 50 # top outer edge
        p5 = 10 # radius corner
        p6 = 25 # x coord hole
        p7 = 25 # y coord hole
        p8 = 5  # Thickness , how to implement this
        self.p = torch.tensor([p1, p2, p3, p4, p5, p6, p7, p8])
        self.h = 2.0 # mesh size parameter
        self.offset = np.array([0.,0.,0.])

        self.circle_radius = 10 
        self.max_V = 1.25e5

        self.logger = None

        const = NonlinearConstraint(
            self.eval_const_numpy,
            -np.inf,
            0.0
        )

        self.constraints = (const,)   

    def set_logger(self, logger):
        self.logger = logger

    def eval_const(self, x):
        """ Evaluates the constraints function g(x). The constraint is fullfilled if g(x) <= 0.

            Args: 
                x (torch.tensor): Vector of design variables

            Returns:
                g (torch.tensor): Constraint function value
        """
        v = volume_bracket(x, self.p, self.circle_radius) 
        g = v - self.max_V       
        return g

    def eval(self, x, xi):
        x = torch.atleast_2d(x)
        xi = torch.atleast_2d(xi)
        Y = []

        for sample_x, sample_xi in zip(x, xi):
            if self.logger is not None: self.logger.info(f"   Evaluate model at x = {sample_x} xi = {sample_xi}")
            compl = sim_bracket_qoi(x=sample_x.tolist(), p=self.p.tolist(), xi=sample_xi.tolist(), circle_radius=self.circle_radius, h=self.h, offset=self.offset)
            Y.append(compl)

        Y_torch = torch.tensor(Y)
        # Y_torch = torch.rand(xi.shape[0])

        return Y_torch.unsqueeze(-1)
