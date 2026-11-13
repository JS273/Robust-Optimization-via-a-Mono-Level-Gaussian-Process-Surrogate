"""Brute-force reference computation.

This module provides functionality for robust optimization using a
brute-force approach. It combines differential evolution for optimization
with Monte Carlo sampling for uncertainty quantification.

The module is used by ``run/brute_force_reference.py``.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

from dataclasses import dataclass, asdict
from scipy.optimize import differential_evolution
import torch 
import numpy as np 
from pathlib import Path
import time

@dataclass(frozen=True)
class ReferenceSpec:
    measure: str
    alpha: float
    tau: float
    n_samples: int
    n_points: int = 100
    popsize: int = 15
    tol: float = 0.01
    maxiter: int = 100
    seed: int = 0
    method: str = "mc"

    def to_dict(self):
        return asdict(self)

@dataclass
class ReferenceResult:
    model_metadata: dict
    spec: dict
    x_star: torch.Tensor
    q_star: torch.Tensor
    oat_plot: dict | None
    domain_plot: dict | None
    opt_info: dict | None
    created_at: float

class ReferenceOpt:
    def __init__(self, root, device="cpu", logger=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.logger = logger

    def _tau_str(self, tau):
        return str(tau).replace(".", "p")

    def result_file(self, model, spec):
        if spec.measure == "quantile":
            tau_str = self._tau_str(spec.tau)
            filepath = self.root / f"{model.name}_tau{tau_str}.pt"
        elif spec.measure == "mean_std":
            alpha_str = self._tau_str(spec.alpha)
            filepath = self.root / f"{model.name}_alpha{alpha_str}.pt"
        return filepath

    def log_file_stem(self, model, spec):
        tau_str = self._tau_str(spec.tau)
        return f"{model.name}_tau{tau_str}"

    def compute_quantile(self, model, x, tau, n_samples):
        x = torch.atleast_2d(x)
        Xi = model.xi_dist.sample(n_samples)
        if x.shape[0] == 1:
            x = x.repeat(n_samples, 1)

        y = model.eval(x, Xi).to(self.device)
        q = torch.quantile(y, tau, dim=0)
        mean = torch.mean(y, dim=0)
        return q.cpu(), mean.cpu()

    def compute_weighted_mean_std(self, model, x, alpha, normalization, n_samples):
        x = torch.atleast_2d(x)
        Xi = model.xi_dist.sample(n_samples)
        if x.shape[0] == 1:
            x = x.repeat(n_samples, 1)

        y = model.eval(x, Xi).to(self.device)
        J = (1-alpha)/normalization[0] * torch.mean(y, dim = 0) + alpha/normalization[1] * torch.std(y, dim = 0)
        mean = torch.mean(y, dim=0)
        return J.cpu(), mean.cpu()

    def get_domain_plot(self, model, spec, normalization):
        if model.n_x > 2:
            return None

        if self.logger:
            self.logger.info("Start domain plot computation")

        x_grid = model.get_xgrid(spec.n_points)
        J_vec = torch.zeros(x_grid.size, dtype=torch.float64)

        for i, x in enumerate(x_grid.nodes):
            if spec.measure == "mean_std":
                J, _ = self.compute_weighted_mean_std(model, torch.from_numpy(x), spec.alpha, normalization, spec.n_samples)
            elif spec.measure == "quantile":
                J, _ = self.compute_quantile(model, torch.from_numpy(x), spec.tau, spec.n_samples)

            J_vec[i] = J.squeeze()


        if self.logger:
            self.logger.info("Finished domain plot computation")

        return {
            "x": x_grid,
            "J": J_vec
        }

    def get_oat_plot(self, model, spec, x_star, q_star, normalization):
        if model.n_x == 1:
            return None

        if self.logger:
            self.logger.info("Start OAT plot computation")

        n_sweep = spec.n_points
        J_sweeps = []

        for i in range(model.n_x):
            x_sweep = torch.linspace(model.x_bounds[i][0], model.x_bounds[i][1], n_sweep)
            x_rep = x_star.repeat(n_sweep, 1)
            x_rep[:, i] = x_sweep

            J_vec = torch.zeros(n_sweep, dtype=torch.float64)

            for j, x in enumerate(x_rep):
                if spec.measure == "mean_std":
                    J, _ = self.compute_weighted_mean_std(model, x, spec.alpha, normalization, spec.n_samples)
                elif spec.measure == "quantile":
                    J, _ = self.compute_quantile(model, x, spec.tau, spec.n_samples)
                J_vec[j] = J.squeeze()

            J_sweeps.append(J_vec)

        if self.logger:
            self.logger.info("Finished OAT plot computation")

        return {
            "J_sweeps": J_sweeps,
            "x_star_ref": x_star,
            "q_star_ref": q_star,
        }

    def optimize(self, model, spec, normalization):
        torch.manual_seed(spec.seed)
        np.random.seed(spec.seed)

        bounds = model.x_bounds.tolist() if hasattr(model.x_bounds, "tolist") else model.x_bounds
        eval_counter = {"n": 0}
        best = {"x": None, "f": np.inf}

        def objfun(x_np):
            eval_counter["n"] += 1
            x = torch.tensor(x_np, dtype=torch.float64)
            if spec.measure == "mean_std":
                J, _ = self.compute_weighted_mean_std(model, x, spec.alpha, normalization, spec.n_samples)
            elif spec.measure == "quantile":
                J, _ = self.compute_quantile(model, x, spec.tau, spec.n_samples)
            val = float(J.squeeze().item())

            if val < best["f"]:
                best["f"] = val
                best["x"] = np.copy(x_np)

            if self.logger and eval_counter["n"] % 20 == 0:
                self.logger.debug(f"DE eval {eval_counter['n']}: x={x_np}, q={val}")

            return val

        def callback(xk, convergence):
            if self.logger:
                self.logger.info(
                    f"DE iter: x_best={xk}, "
                    f"f_best={best['f']:.6e}, "
                    f"convergence={convergence:.6e}, "
                    f"n_eval={eval_counter['n']}"
                )
            return False

        if self.logger:
            self.logger.info("Start differential evolution")
            self.logger.info(
                f"DE settings: maxiter={spec.maxiter}, popsize={spec.popsize}, "
                f"tol={spec.tol}, seed={spec.seed}"
            )

        result = differential_evolution(
            objfun,
            bounds=bounds,
            maxiter=spec.maxiter,
            popsize=spec.popsize,
            tol=spec.tol,
            polish=False,
            disp=False,
            seed=spec.seed,
            callback=callback,
        )

        if self.logger:
            self.logger.info("Finished differential evolution")
            self.logger.info(f"DE success: {result.success}")
            self.logger.info(f"DE message: {result.message}")
            self.logger.info(f"DE nit: {result.nit}")
            self.logger.info(f"DE nfev: {result.nfev}")
            self.logger.info(f"DE best x: {result.x}")
            self.logger.info(f"DE best f: {result.fun}")

        x_star = torch.tensor(result.x, dtype=torch.float64)
        q_star = torch.tensor([result.fun], dtype=torch.float64)

        opt_info = {
            "success": result.success,
            "message": result.message,
            "nfev": result.nfev,
            "nit": result.nit,
        }
        return x_star, q_star, opt_info

    def run(self, model, spec, normalization = None, x_star=None):
        if self.logger:
            self.logger.info("Start reference computation")

        if x_star is None:
            x_star, q_star, opt_info = self.optimize(model, spec, normalization)
        else:
            if self.logger:
                self.logger.info(f"Using provided x_star = {x_star}")
            q_star, _ = self.compute_quantile(model, x_star, spec.tau, spec.n_samples)
            opt_info = None

        oat_plot = self.get_oat_plot(model, spec, x_star, q_star, normalization)
        domain_plot = self.get_domain_plot(model, spec, normalization)

        if self.logger:
            self.logger.info(f"x_star = {x_star}")
            self.logger.info(f"q_star = {q_star}")
            self.logger.info("Finished reference computation")

        return ReferenceResult(
            model_metadata=model.metadata,
            spec=spec.to_dict(),
            x_star=x_star.cpu(),
            q_star=q_star.cpu(),
            oat_plot=oat_plot,
            domain_plot=domain_plot,
            opt_info=opt_info,
            created_at=time.time(),
        )

    def save(self, model, spec, ref_result):
        file_path = self.result_file(model, spec)
        torch.save(ref_result, file_path)
        if self.logger:
            self.logger.info(f"Saved reference result to: {file_path}")

    def load(self, model, spec):
        return torch.load(self.result_file(model, spec), weights_only=False)
