"""Convergence criteria.

This module provides functionality for checking convergence of the MLRO
approach. During each optimization phase—expected improvement (EI), global
mean optimization (EM), and trust-region optimization (TR)—different
convergence criteria are evaluated, including:

- Monitoring the acquisition-function value.
- Monitoring changes in the robustness measure.
- Monitoring changes in the design vector.

Author
------
Julius Schultz

Date
----
2026-10-16

"""


import json
import torch
import numpy as np 

# Convergence checks
def check_EI_conv(ro_data, logger, args):
    conv_crit = json.loads(args.conv_EI)
    EI_vals = ro_data.EI_infill["EI_val"]
    eps = conv_crit["eps"]

    i_EI = ro_data.EI_infill["n"]
    n = conv_crit["n_ave"]

    break_flag = False

    if (i_EI >= n) & (conv_crit["switch"] == 1):
        mean_EI = np.mean(np.array(EI_vals[-n:]))
        if mean_EI < eps*EI_vals[0]:
            logger.info(f"    Converged: EI/EI_max = {mean_EI/max(EI_vals)} < {eps} --> Convergence criterion met after {i_EI}/{args.n_EI} iterations \n")
            break_flag = True
        else:
            logger.info(f"    Not converged: EI/EI_max = {mean_EI/max(EI_vals)} > {eps} --> Continue EI \n")
    

    return break_flag

def check_conv(logger, i, max_i, val, n_ave, eps_abs, eps_rel, name):
    break_flag = False
    if i >= n_ave+1:
        eps = 1e-12

        curr = val[1:]
        prev = val[:-1]

        abs_deltas = torch.linalg.norm(curr - prev, dim=1)
        val_norms = torch.linalg.norm(val, dim=1)
        scale = torch.maximum(torch.max(val_norms), torch.tensor(eps))

        rel_deltas = abs_deltas / scale

        abs_ma = abs_deltas[-n_ave:].mean().item()
        rel_ma = rel_deltas[-n_ave:].mean().item()

        break_flag = False

        if abs_ma <= eps_abs:
            logger.info(f"    {name}_abs_ma = {abs_ma:.2e} < {eps_abs} --> Converged - criterion met after {i}/{max_i} iterations")
            break_flag = True
        else:
            logger.info(f"    {name}_abs_ma = {abs_ma:.2e} > {eps_abs} --> Continue")

        if rel_ma <= eps_rel:
            logger.info(f"    {name}_rel_ma = {rel_ma:.2e} < {eps_rel} --> Converged - criterion met after {i}/{max_i} iterations\n")
            break_flag = True
        else:
            logger.info(f"    {name}_rel_ma = {rel_ma:.2e} > {eps_rel} --> Continue\n")

    return break_flag

def check_TR_conv(ro_data, logger, args):
    i = ro_data.TR_infill["n"]
    max_i = args.n_TR
    break_flag_x = break_flag_y = False

    # Check convegence in x
    crit_x = json.loads(args.conv_x)
    x_iter = torch.vstack(ro_data.TR_infill["cand_x"])  
    if crit_x["switch"] and (i > crit_x["n_ave"]): break_flag_x = check_conv(logger=logger, i=i, max_i=max_i, val=x_iter, n_ave=crit_x["n_ave"], eps_abs=crit_x["abs_x"], eps_rel=crit_x["rel_x"], name="delta_x")

    # Check convergence in y
    crit_y = json.loads(args.conv_y)
    y_iter = torch.vstack(ro_data.TR_infill["cand_J"])  
    if crit_y["switch"]: break_flag_y = check_conv(logger=logger, i=i, max_i=max_i, val=y_iter, n_ave=crit_y["n_ave"], eps_abs=crit_y["abs_y"], eps_rel=crit_y["rel_y"], name="delta_y")

    return break_flag_x | break_flag_y

def check_EM_conv(ro_data, logger, args):
    i = ro_data.EM_infill["n"]
    max_i = args.n_EM
    break_flag_x = break_flag_y = False

    # Check convegence in x
    crit_x = json.loads(args.conv_x)
    x_iter = torch.vstack(ro_data.EM_infill["cand_x"])  
    if crit_x["switch"]: break_flag_x = check_conv(logger=logger, i=i, max_i=max_i, val=x_iter, n_ave=crit_x["n_ave"], eps_abs=crit_x["abs_x"], eps_rel=crit_x["rel_x"], name="delta_x")

    # Check convergence in y
    crit_y = json.loads(args.conv_y)
    y_iter = torch.vstack(ro_data.EM_infill["cand_J"])  
    if crit_y["switch"]: break_flag_y = check_conv(logger=logger, i=i, max_i=max_i, val=y_iter, n_ave=crit_y["n_ave"], eps_abs=crit_y["abs_y"], eps_rel=crit_y["rel_y"], name="delta_y")

    return break_flag_x | break_flag_y

