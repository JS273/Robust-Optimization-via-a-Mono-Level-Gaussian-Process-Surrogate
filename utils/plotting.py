"""Plotting utilities for robust optimization and surrogate-model analysis.

This module provides functions for visualizing objective landscapes,
candidate and best-design evolution, convergence behavior, acquisition
functions, trust regions, and uncertainty in robustness estimates. It also
generates domain, one-at-a-time sweep, and infill plot data for diagnostic
analysis during the MLRO workflow.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch, time
import seaborn as sns
from ro.acfn import RobustnessMeasure
from ro.helper import get_matherons_path
ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

def contour_plot(grid, values, ax = None, cbar = False, cmap = "viridis", cbar_ticks = None, levels = 30, **kwargs):
    xx = grid.meshgrid_mat[0]
    yy = grid.meshgrid_mat[1]

    if values.ndim == 2:
        zz = values
    else:
        zz = grid.reshape_data(values)

    if ax is None:
        fig, ax = plt.subplots()

    ax.grid(False)
    cs = ax.contourf(xx,yy,zz, cmap = cmap, levels = levels, **kwargs)

    cs.set_edgecolors("face")

    if cbar:
        if cbar_ticks: cbar = plt.colorbar(cs, ax = ax, ticks = cbar_ticks)
        else: cbar = plt.colorbar(cs, ax = ax)

    return ax

def add_double_arrow(ax, start_x, end_x, y_position, label, shift_text_y = 0.02):
    # Create a double arrow to represent the interval
    arrow_length = 0.2  # Adjust this for arrow length
    arrow = FancyArrowPatch((start_x, y_position), (end_x, y_position),
                             arrowstyle='<|-|>', color='black', lw=1,
                             mutation_scale=10, shrinkA= 0.0, shrinkB= 0.0)
    ax.add_patch(arrow)
    
    # Add the label in the middle of the arrow
    ax.text((start_x + end_x) / 2, y_position + shift_text_y, label, ha='center', va='bottom', fontsize=8)

    return ax

def add_iteration_arrows(ax, model_calls_cumsum, n_EI, n_EM, ypos = None, ymin = None, shift_text_y = 0.04):
    ylim = ax.get_ylim()
    delta_y = ylim[1]-ylim[0]
    shift = 0.075

    if ypos is None:
        ypos = ylim[0]-shift*delta_y

    ax = add_double_arrow(ax = ax, start_x=model_calls_cumsum[0], end_x=model_calls_cumsum[n_EI], y_position=ypos, label = "EI", shift_text_y = shift_text_y)
    ax = add_double_arrow(ax = ax, start_x=model_calls_cumsum[n_EI], end_x=model_calls_cumsum[n_EI+n_EM], y_position=ypos, label = "EM", shift_text_y = shift_text_y)
    ax = add_double_arrow(ax = ax, start_x=model_calls_cumsum[n_EI+n_EM], end_x=model_calls_cumsum[-1], y_position=ypos, label = "TR", shift_text_y = shift_text_y)
    
    if ymin is None:
        ax.set_ylim([ylim[0]-1.5*shift*delta_y, ylim[1]])
    else:
        ax.set_ylim([ymin, ylim[1]])
    
    return ax

def add_vline(ax, n_doe, n_EI, n_EM, n_TR):
    ax.axvline(x=n_doe, linestyle="-.", linewidth=1, color = "gray")
    ax.axvline(x=n_EI, linestyle="-.", linewidth=1, color = "gray")
    ax.axvline(x=n_EM, linestyle="-.", linewidth=1, color = "gray")
    ax.axvline(x=n_TR, linestyle="-.", linewidth=1, color = "gray")
    
def create_oat_plot(sweep_plots, oat_plot, J_star):

    n_dim = len(sweep_plots["J_mean_sweeps"])
    n_row = int(np.floor(np.sqrt(n_dim)))
    n_col = int(np.ceil(n_dim/n_row))
    fig, ax = plt.subplots(n_row,n_col, figsize = [5,3], sharey = True)
    
    for idx, ax in enumerate(ax.flatten()): 
        ax.plot(oat_plot["x_sweep"], oat_plot["J_sweeps"][idx])
        ax.plot(sweep_plots["x_sweep"], sweep_plots["J_mean_sweeps"][idx])
        upper_bound = sweep_plots["J_mean_sweeps"][idx] + sweep_plots["J_std_sweeps"][idx]
        lower_bound = sweep_plots["J_mean_sweeps"][idx] - sweep_plots["J_std_sweeps"][idx]
        ax.fill_between(sweep_plots["x_sweep"], upper_bound, lower_bound, color=ccycle[0], alpha=0.3, zorder = 50)
        ax.axhline(J_star)
        ax.set_xlabel(f"x{idx}")

def plot_cand_evolution(ro_data, model, args, output_folder = None, tag = None):
    plot_ref = True
    try:
        ref_x, ref_q = model.load_opt(measure = args.measure, tau = args.tau, alpha = args.alpha)
        ref_x = np.atleast_1d(ref_x)
    except:
        plot_ref = False
    cand_x_list =  [ro_data.doe["cand_x"]] + ro_data.EI_infill["cand_x"] + ro_data.EM_infill["cand_x"] + ro_data.TR_infill["cand_x"]
    cand_J_list = [ro_data.doe["cand_J"]] + ro_data.EI_infill["cand_J"] + ro_data.EM_infill["cand_J"] + ro_data.TR_infill["cand_J"]

    model_calls = ro_data.doe["model_calls"] + ro_data.EI_infill["model_calls"] + ro_data.EM_infill["model_calls"] + ro_data.TR_infill["model_calls"]

    cand_x = np.array(cand_x_list)
    cand_J = np.array(cand_J_list)
    model_calls_cumsum = np.cumsum(np.array(model_calls))
    n_EI = ro_data.EI_infill["n"]
    n_EM = ro_data.EM_infill["n"]
    n_TR = ro_data.TR_infill["n"]

    fig, ax = plt.subplots(1,2, figsize = [6,3])
    ax[0].plot(model_calls_cumsum, cand_J)
    ax[0].set_xlabel("Model calls")
    ax[0].set_ylabel("Quantile")
    ax[0].set_title("Candiate evolution")
    if plot_ref: ax[0].plot(model_calls_cumsum, np.ones(len(model_calls_cumsum))*ref_q.numpy(), linestyle = "--", color = "grey")
    
    ax[0] = add_iteration_arrows(ax[0], model_calls_cumsum, n_EI, n_EM)

    ax[1].plot(model_calls_cumsum, cand_x)
    ax[1].set_xlabel("Model calls")
    ax[1].set_ylabel("Design point")
    ax[1].set_title("Candiate evolution")
    if plot_ref: ax[1].plot(model_calls_cumsum, np.ones((len(model_calls_cumsum), len(ref_x)))*ref_x, linestyle = "--", color = "grey")

    ax[1] = add_iteration_arrows(ax[1], model_calls_cumsum, n_EI, n_EM)
    fig.tight_layout()
    
    if output_folder is not None:
        file_name = tag + '_cand_evo'
        fig.tight_layout()
        fig.savefig(f'{output_folder}/{file_name}.pdf', format = 'pdf', bbox_inches = 'tight', pad_inches = 0.0)

def plot_best_evolution(
    ro_data,
    model,
    args,
    output_folder=None,
    tag=None,
    fig_ax=None,
    continuous_error=True
):
    
    plot_ref = True
    try:
        ref_x, ref_q = model.load_opt(measure = args.measure, tau = args.tau, alpha = args.alpha)
        ref_x = np.atleast_1d(ref_x)
    except Exception:
        plot_ref = False

    best_x_list = [ro_data.doe["cand_x"]] + ro_data.EI_infill["best_x"] + ro_data.EM_infill["best_x"] + ro_data.TR_infill["best_x"]
    best_J_list = [ro_data.doe["cand_J"]] + ro_data.EI_infill["best_J"] + ro_data.EM_infill["best_J"] + ro_data.TR_infill["best_J"]
    best_J_std_list = [ro_data.doe["cand_J_std"]] + ro_data.EI_infill["best_J_std"] + ro_data.EM_infill["best_J_std"]

    model_calls = ro_data.doe["model_calls"] + ro_data.EI_infill["model_calls"] + ro_data.EM_infill["model_calls"] + ro_data.TR_infill["model_calls"]

    best_x = np.array(best_x_list)
    best_J = np.array(best_J_list)
    best_J_std = np.array(best_J_std_list)
    model_calls_cumsum = np.cumsum(np.array(model_calls))

    n_EI = ro_data.EI_infill["n"]
    n_EM = ro_data.EM_infill["n"]
    n_TR = ro_data.TR_infill["n"]

    created_fig = False
    if fig_ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        created_fig = True
    else:
        fig, ax = fig_ax

    J_line, = ax[0].plot(model_calls_cumsum, best_J, label="Best $J$")

    x_err = model_calls_cumsum[:n_EI + n_EM + 1]
    y_err = best_J[:n_EI + n_EM + 1]
    y_std = best_J_std[:n_EI + n_EM + 1]

    if continuous_error:
        ax[0].fill_between(
            x_err,
            y_err - y_std,
            y_err + y_std,
            alpha=0.3,
            color=J_line.get_color(),
            label=r"$\pm$ std"
        )
    else:
        ax[0].errorbar(
            x_err,
            y_err,
            y_std,
            fmt='o',
            capsize=5,
            color=J_line.get_color(),
            label='Std'
        )

    if plot_ref:
        ax[0].plot(
            model_calls_cumsum,
            np.ones(len(model_calls_cumsum)) * ref_q.numpy(),
            linestyle="--",
            color="grey",
            label="Ref. $J$"
        )

    ax[0].set_xlabel("Model calls")
    ax[0].set_ylabel("Quantile")
    ax[0] = add_iteration_arrows(ax[0], model_calls_cumsum, n_EI, n_EM)
    ax[0].legend()

    labels = [f"$x_{i}$" for i in range(1, best_x.shape[1] + 1)]

    lines = ax[1].plot(model_calls_cumsum, best_x)
    for line, label in zip(lines, labels):
        line.set_label(label)

    if plot_ref:
        for i, line in enumerate(lines):
            ax[1].plot(
                model_calls_cumsum,
                np.ones(len(model_calls_cumsum)) * ref_x[i],
                linestyle="--",
                color=line.get_color()
            )

    ax[1].set_xlabel("Model calls")
    ax[1].set_ylabel("Design point")
    ax[1] = add_iteration_arrows(ax[1], model_calls_cumsum, n_EI, n_EM)
    ax[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

    sns.despine(ax=ax[0])
    sns.despine(ax=ax[1])

    if created_fig:
        fig.tight_layout()

    if output_folder is not None:
        file_name = tag + '_best_evo'
        fig.tight_layout()
        fig.savefig(
            f'{output_folder}/{file_name}.pdf',
            format='pdf',
            bbox_inches='tight',
            pad_inches=0.0
        )

    return fig, ax

def plot_convergence(
    ro_data,
    model,
    args,
    output_folder=None,
    tag=None,
    fig_ax=None
):
    
    ref_available = True
    try:
        ref_x, ref_J = model.load_opt(measure = args.measure, tau = args.tau, alpha = args.alpha)
        ref_x = torch.atleast_1d(ref_x)
    except Exception:
        ref_available = False

    if not ref_available:
        return

    best_x_list = [ro_data.doe["cand_x"]] + ro_data.EI_infill["best_x"] + ro_data.EM_infill["best_x"] + ro_data.TR_infill["best_x"]
    best_J_list = [ro_data.doe["cand_J"]] + ro_data.EI_infill["best_J"] + ro_data.EM_infill["best_J"] + ro_data.TR_infill["best_J"]
    model_calls = ro_data.doe["model_calls"] + ro_data.EI_infill["model_calls"] + ro_data.EM_infill["model_calls"] + ro_data.TR_infill["model_calls"]

    best_x = torch.row_stack(best_x_list)
    best_J = torch.tensor(best_J_list)
    model_calls_cumsum = torch.cumsum(torch.tensor(model_calls), dim = 0)

    n_EI = ro_data.EI_infill["n"]
    n_EM = ro_data.EM_infill["n"]
    n_TR = ro_data.TR_infill["n"]

    created_fig = False
    if fig_ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        created_fig = True
    else:
        fig, ax = fig_ax

    best_J_rel = torch.abs(best_J - ref_J)/torch.abs(ref_J)
   
    true_norm = torch.linalg.norm(ref_x).clamp_min(1e-12)      # scalar
    best_x_rel = torch.linalg.norm(best_x - ref_x, dim=-1) / true_norm

    ax[0].set_title("Convergence in J")
    ax[0].semilogy(model_calls_cumsum, best_J_rel)
    ax[0].grid(True, alpha = 0.3)
    ax[0].set_ylim([1e-4, 1e-0])
    ax[0] = add_iteration_arrows(ax[0], model_calls_cumsum, n_EI, n_EM, ypos= 1.5e-4, ymin = 1e-4, shift_text_y= 0)

    ax[1].set_title("Convergence in x")
    ax[1].semilogy(model_calls_cumsum, best_x_rel)
    ax[1].grid(True, alpha = 0.3)
    ax[0].set_ylim([1e-4, 1e-0])
    ax[1] = add_iteration_arrows(ax[1], model_calls_cumsum, n_EI, n_EM, ypos= 1.5e-4, ymin = 1e-4, shift_text_y= 0)

    sns.despine(ax=ax[0])
    sns.despine(ax=ax[1])

    if created_fig:
        fig.tight_layout()

    if output_folder is not None:
        file_name = tag + '_convergence'
        fig.tight_layout()
        fig.savefig(
            f'{output_folder}/{file_name}.pdf',
            format='pdf',
            bbox_inches='tight',
            pad_inches=0.0
        )

    return fig, ax

def add_DOE_plot(ro_data, model, gp_model, J_sampler, args):
    
    # Domain Plot
    if model.n_x <= 2:
        x_domain = model.get_xgrid(n_points= args.plot_n_points)
        J_mean_vec = torch.zeros(x_domain.size)
        J_std_vec = torch.zeros(x_domain.size)

        if J_sampler.sampler == "matheron":
            path = get_matherons_path(gp_model, args.plot_n_samp_gp, J_sampler.J_device)
        else:
            path = None

        for idx, x_c in enumerate(x_domain.nodes):
            J_mean_vec[idx], J_std_vec[idx] = J_sampler.get_J_sample_moments(torch.from_numpy(x_c), gp_model, args.plot_n_samp_J, args.plot_n_samp_gp, path)

    else:
        x_domain, J_mean_vec, J_std_vec = None, None, None

    # OAT Plot
    if model.n_x >= 2:
        sweep_plots = get_sweep_plot(model=model, gp_model=gp_model, J_sampler=J_sampler, args=args)
    else:
        sweep_plots = None
    
    plot_data = {"x_domain": x_domain,
                "J_mean_vec": J_mean_vec,
                "J_std_vec": J_std_vec,
                "tau": args.tau,
                "bounds": model.x_bounds,
                "sweep_plots":sweep_plots
                }
    
    ro_data.append_doe_plot(plot_data)

def add_DOE_infill_plot(ro_data, model, gp_model, J_sampler: RobustnessMeasure, args):
    
    # Domain Plot
    if model.n_x <= 2:
        x_domain = model.get_xgrid(n_points= args.plot_n_points)
        J_mean_vec = torch.zeros(x_domain.size)
        J_std_vec = torch.zeros(x_domain.size)

        if J_sampler.sampler == "matheron":
            path = get_matherons_path(gp_model, args.plot_n_samp_gp, J_sampler.J_device)
        else:
            path = None

        for idx, x_c in enumerate(x_domain.nodes):
            J_mean_vec[idx], J_std_vec[idx] = J_sampler.get_J_sample_moments(torch.from_numpy(x_c), gp_model, args.plot_n_samp_J, args.plot_n_samp_gp, path)

    else:
        x_domain, J_mean_vec, J_std_vec = None, None, None

    # OAT Plot
    if model.n_x >= 2:
        sweep_plots = get_sweep_plot(model=model, gp_model=gp_model, J_sampler=J_sampler, args=args)
    else:
        sweep_plots = None
    
    plot_data = {"x_domain": x_domain,
                "J_mean_vec": J_mean_vec,
                "J_std_vec": J_std_vec,
                "tau": args.tau,
                "bounds": model.x_bounds,
                "sweep_plots":sweep_plots
                }
    
    ro_data.append_doe_infill_plot(plot_data)
                            
def add_EI_plot(ro_data, model, acfn, x_c, x_star, J_star, args):
    
    # Domain Plot
    if model.n_x <= 2:
        x_domain = model.get_xgrid(n_points= args.plot_n_points)
        JEI, JEI_samp, LogJEI, J_mean_vec, J_std_vec = acfn.sweep_obj(x_vec= x_domain, n_samp_J= args.plot_n_samp_J, n_samp_gp= args.plot_n_samp_gp, n_rep = args.plot_n_rep)
    else:
        x_domain, JEI, J_mean_vec, J_std_vec = None, None, None, None

    # OAT Plot
    if model.n_x >= 2:
        sweep_plots = get_sweep_plot(model=model, gp_model=acfn.gp_model, J_sampler=acfn.J_sampler, args=args, J_star=J_star)
    else:
        sweep_plots = None

    plot_data = {"x_domain": x_domain,
                    "J_mean_vec": J_mean_vec,
                    "J_std_vec": J_std_vec,
                    "tau": args.tau,
                    "bounds": model.x_bounds,
                    "J_star": J_star,
                    "x_star": x_star,
                    "JEI": JEI,
                    "JEI_sampling": JEI_samp,
                    "LogJEI": LogJEI,
                    "x_c": x_c,
                    "EI_star": ro_data.EI_infill["EI_val"][-1],
                    "gp_model": acfn.gp_model, 
                    "sweep_plots": sweep_plots
                }
    
    ro_data.append_EI_plot(plot_data)

def add_EM_plot(ro_data, model, gp_model, x_c, J_c, x_star, J_star, J_sampler, args, logger):
    t1 = time.time()
    # Domain Plot
    if model.n_x <= 2:
        x_domain = model.get_xgrid(n_points= args.plot_n_points)
        J_vec = torch.zeros((x_domain.size, args.plot_n_rep))

        if J_sampler.sampler == "matheron":
            path = get_matherons_path(gp_model, args.plot_n_samp_gp, J_sampler.J_device)
        else:
            path = None

        for i in range(args.plot_n_rep):
            for idx,x in enumerate(x_domain.nodes):
                J_vec[idx, i] = torch.mean(J_sampler.get_J_sample(torch.from_numpy(x), gp_model, args.plot_n_samp_J, args.plot_n_samp_gp, path))
    else: 
        x_domain, J_vec = None, None

    # OAT Plot
    if model.n_x >= 2:
        sweep_plots = get_sweep_plot(model=model, gp_model=gp_model, J_sampler=J_sampler, args=args, J_star=J_star)
    else:
        sweep_plots = None

    plot_data = {"x_domain": x_domain,
                    "J_vec": J_vec,
                    "tau": args.tau,
                    "bounds": model.x_bounds,
                    "J_star": J_star,
                    "x_star": x_star,
                    "x_c": x_c,
                    "J_c": J_c,
                    "gp_model": gp_model,
                    "sweep_plots": sweep_plots
                }
    ro_data.append_EM_plot(plot_data)
    logger.debug(f"     Timing: Gen Plot data: t = {time.time() - t1:.3f} s")

def add_TR_plot(ro_data, model, gp_model, x_c, J_c, x_star, J_star, trust_reg, J_sampler: RobustnessMeasure, args):
    if model.n_x <= 2:
        x_domain = model.get_xgrid(n_points= args.plot_n_points)
        J = torch.zeros(x_domain.size)

        for idx, x in enumerate(x_domain.nodes):
            J[idx] = J_sampler.get_gp_mean_J(x = torch.from_numpy(x), gp_model = gp_model, n_samp_J= args.n_samp_J)
    else: 
        x_domain, J = None, None

    # OAT Plot
    if model.n_x >= 2:
        sweep_plots = get_sweep_plot(model=model, gp_model=gp_model, J_sampler=J_sampler, args=args, J_star=J_star)
    else:
        sweep_plots = None

    plot_data = {"x_domain": x_domain,
                    "J": J,
                    "tau": args.tau,
                    "bounds": model.x_bounds,
                    "J_star": J_star,
                    "x_star": x_star,
                    "x_c": x_c,
                    "J_c": J_c,
                    "trust_reg": trust_reg,
                    "gp_model": gp_model,
                    "sweep_plots": sweep_plots
                }
    ro_data.append_TR_plot(plot_data)

def get_sweep_plot(model, gp_model, J_sampler, args, J_star = None):
    x_star_ref, opt_q = model.load_opt(measure = args.measure, tau = args.tau, alpha = args.alpha)

    n_sweep = 30
    
    J_mean_sweeps = []
    J_std_sweeps = []

    for i in range(model.n_x):
        x_sweep = torch.linspace(model.x_bounds[i][0], model.x_bounds[i][1], n_sweep)
        x_rep = x_star_ref.repeat(n_sweep, 1)
        x_rep[:,i] = x_sweep

        path = get_matherons_path(gp_model, args.plot_n_samp_gp, J_sampler.J_device)
        J_mean_vec = torch.zeros(x_rep.shape[0])
        J_std_vec = torch.zeros(x_rep.shape[0])

        for idx,x in enumerate(x_rep):
            J_mean_vec[idx], J_std_vec[idx] = J_sampler.get_J_sample_moments(x, gp_model, args.plot_n_samp_J, args.plot_n_samp_gp, path)
            
        J_mean_sweeps.append(J_mean_vec)
        J_std_sweeps.append(J_std_vec)

    plot_data = {"x_sweep": x_sweep,
                "J_mean_sweeps": J_mean_sweeps,
                "J_std_sweeps": J_std_sweeps,
                "J_star": J_star}
    
    return plot_data