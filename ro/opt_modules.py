"""Core implementation of the mono-level robust optimization algorithm.

This module provides the main workflow for the different stages of the MLRO
approach, including:

- Initial design-of-experiments (DOE) generation.
- DOE refinement.
- Expected-improvement (EI) infill.
- Expected-mean (EM) infill.
- Trust-region (TR) infill.

Each stage selects new infill points using the appropriate acquisition
function, updates the mono-level Gaussian-process model, and performs the
associated logging, plotting, and data-storage tasks.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

import numpy as np
import torch, time
import gc
from scipy.optimize import differential_evolution
from utils.logger import log_doe, log_infill, log_infill_TR, log_infill_EM, get_cuda_memory
from utils.datastructures import save_ro_data
from ro.helper import get_gp_model, copy_gp_object
from ro.acfn import AcfnJEI, AcfnUQ, AcfnTR, RobustnessMeasure, comp_TR_radius, AcfnEM
from ro.convergence import check_EI_conv, check_EM_conv, check_TR_conv
from utils.plotting import add_DOE_plot, add_DOE_infill_plot, add_EI_plot, add_EM_plot, add_TR_plot
torch.set_default_dtype(torch.float64)

def opt_stochastic_domain_acfn(model, acfn_uq, args, se_infill, logger):
    z_bounds = model.get_stoch_box_domain(acfn_uq.x)
    tstart = time.time() 
    if se_infill:
        opt_uq = differential_evolution(acfn_uq.obj_fun_SE_np, bounds = z_bounds.numpy(), maxiter= args.UQ_genopt_i, popsize = args.UQ_genopt_pop, polish= True, vectorized = True, updating='deferred', tol = 0.005)
        logger.debug(f"     Timing: SE Opt: t = {time.time() - tstart:.3f} s")
        logger.debug(f"     SE Opt: nit={opt_uq.nit}, nfev={opt_uq.nfev}, message={opt_uq.message}")
    else:
        opt_uq = differential_evolution(acfn_uq.obj_fun_UFun_np, bounds = z_bounds.numpy(), maxiter= args.UQ_genopt_i, popsize = args.UQ_genopt_pop, polish= True, vectorized = True, updating='deferred', tol = 0.005)
        logger.debug(f"     Timing: UFun Opt: t = {time.time() - tstart:.3f} s")
        logger.debug(f"     Ufun Opt: nit={opt_uq.nit}, nfev={opt_uq.nfev}, message={opt_uq.message}")
    z_star = torch.from_numpy(opt_uq.x)
    return z_star

def create_init_gp(model, J_sampler: RobustnessMeasure, ro_data, args, logger = None):
    if logger is not None: logger.info(f"Initial DOE:")
    if args.load_doe and not args.transfer_learning:
        if logger is not None:  logger.info(f"Load doe")
        n_doe_loaded, load_sobol_burn_in = ro_data.load_doe(args.sim_file)
        if args.continue_doe:
            sobol_burn_in = load_sobol_burn_in + n_doe_loaded
        else:
            gp_model = get_gp_model(ro_data = ro_data, x_map = model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)
            if args.measure == "mean_std": J_sampler.get_norm_mean_std()
            return gp_model, ro_data, J_sampler
    elif args.transfer_learning:
        logger.info(f"  Transfer learning: Load simulation data to DOE")
        ro_data.reuse_sim(args, logger)
        gp_model = get_gp_model(ro_data = ro_data, x_map = model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)
        if args.measure == "mean_std": J_sampler.get_norm_mean_std()
        return gp_model, ro_data, J_sampler

    else:
        sobol_burn_in = args.doe_burn_in
    
    sobol_engine = torch.quasirandom.SobolEngine(dimension=model.n_dim, scramble=True, seed = args.seed)
    sobol_engine.fast_forward(sobol_burn_in)
    init_train_u = sobol_engine.draw(n=args.n_doe)
    init_train_xtilde = model.xtilde_map_det_design.untransform(init_train_u)
    init_train_y = model.eval(init_train_xtilde[:,:model.n_x], init_train_xtilde[:,model.n_x:])
    if args.continue_doe:
        ro_data.append_doe(init_train_xtilde, init_train_y)
    else:
        ro_data.set_doe(init_train_xtilde, init_train_y, sobol_burn_in)

    if args.measure == "mean_std": J_sampler.get_norm_mean_std()

    gp_model = get_gp_model(ro_data = ro_data, x_map = model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)

    # Get the current optimum
    opti_J = AcfnEM(gp_model, J_sampler, args)
    result = differential_evolution(opti_J.obj_fun_min_J, args = (args.n_samp_J_opt, args.n_samp_gp_opt), bounds = model.x_bounds, maxiter= args.EI_genopt_i, popsize = args.EI_genopt_pop, polish= False, constraints=model.constraints)
    x_glob = torch.from_numpy(result.x)
    J_glob, J_glob_std = J_sampler.get_J_sample_moments(x=x_glob, gp_model = gp_model, n_samp_J=args.n_samp_J, n_samp_gp=args.n_samp_gp)
    ro_data.doe["cand_x"], ro_data.doe["cand_J"], ro_data.doe["cand_J_std"] = x_glob, J_glob, J_glob_std 
    # ro_data.doe["cand_x"], ro_data.doe["cand_J"], ro_data.doe["cand_J_std"], _ = J_sampler.get_min_J(gp_model, model, ro_data.doe["xtilde"][:,:model.n_x], args.n_samp_gp, args.n_samp_J)
    ro_data.doe["model_calls"].append(args.n_doe)
    if args.plot_flag: add_DOE_plot(ro_data=ro_data, model=model, gp_model=gp_model, J_sampler=J_sampler, args=args)
   
    if logger is not None:  log_doe(logger, model, ro_data, init_train_y, init_train_xtilde, args)
        
    return gp_model, ro_data, J_sampler

def perform_doe_infill(gp_model, model, J_sampler: RobustnessMeasure, ro_data, logger, args):
    logger.info(f"DOE Infill:")
    if args.load_doe_infill and not args.transfer_learning:
        if logger is not None:  logger.info(f"Load doe infill")
        n_doe_infill_loaded = ro_data.load_doe_infill(args.sim_file)
        gp_model = get_gp_model(ro_data = ro_data, x_map = model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)
        if args.catch_up_doe_infill:
            logger.info(f"  Catch up doe infill from sample {n_doe_infill_loaded+1}")
            start_idx = n_doe_infill_loaded
        elif args.continue_doe_infill:
            logger.info(f"  Continue doe infill for all samples")
            start_idx = 0
            ro_data.doe_infill["n"] = 0
        else:
            return gp_model, ro_data
    else:
        ro_data.doe_infill["n"] = 0
        start_idx = 0
        ro_data.clear_doe_infill()
    
    for i, x_c in enumerate(ro_data.doe["xtilde"][start_idx:,:model.n_x]):
        logger.info(f"  Iter {start_idx + i+1} / {ro_data.doe['n']}:")
        J_c = torch.mean(J_sampler.get_J_sample(x_c, gp_model, args.n_samp_J, args.n_samp_gp))

        for j in range(args.n_doe_se + args.n_doe_ufun):
            # Acquisition optimization
            acfn_uq = AcfnUQ(gp_model, x = x_c, model = model, J = J_c, measure = args.measure)
            z_star = opt_stochastic_domain_acfn(model = model, acfn_uq = acfn_uq, args = args, se_infill= (j+1) <= args.n_doe_se, logger = logger)

            # Observe
            xtilde = model.get_xtilde(x_c, z_star)
            new_y = model.eval_xtilde(xtilde)
            
            # Concatenate
            ro_data.append_doe_infill(xtilde, new_y)
            gp_model = get_gp_model(ro_data, model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)

            J_c = torch.mean(J_sampler.get_J_sample(x_c, gp_model, args.n_samp_J, args.n_samp_gp))
        ro_data.doe["model_calls"][0] += args.n_doe_se + args.n_doe_ufun
        ro_data.doe_infill["n"] = ro_data.doe_infill["n"] + 1
        save_ro_data(ro_data=ro_data, model=model, gp_model=gp_model, args=args)

    # Get the current opt
    opti_J = AcfnEM(gp_model, J_sampler, args)
    result = differential_evolution(opti_J.obj_fun_min_J, args = (args.n_samp_J_opt, args.n_samp_gp_opt), bounds = model.x_bounds, maxiter= args.EI_genopt_i, popsize = args.EI_genopt_pop, polish= False, constraints=model.constraints)
    x_glob = torch.from_numpy(result.x)
    J_glob, J_glob_std = J_sampler.get_J_sample_moments(x=x_glob, gp_model = gp_model, n_samp_J=args.n_samp_J, n_samp_gp=args.n_samp_gp)
    ro_data.doe["cand_x"], ro_data.doe["cand_J"], ro_data.doe["cand_J_std"] = x_glob, J_glob, J_glob_std 
    # ro_data.doe["cand_x"], ro_data.doe["cand_J"], ro_data.doe["cand_J_std"], _ = J_sampler.get_min_J(gp_model, model, ro_data.doe["xtilde"][:,:model.n_x], args.n_samp_gp, args.n_samp_J)
    
    if args.plot_flag: add_DOE_infill_plot(ro_data=ro_data, model=model, gp_model=gp_model, J_sampler=J_sampler, args=args)
    return gp_model, ro_data

def perform_EI_infill(gp_model, model, J_sampler: RobustnessMeasure, ro_data, logger, args):
    if args.load_EI and not args.transfer_learning:
        if logger is not None:  logger.info(f"Load EI infill")
        ro_data.load_EI(args.sim_file)
        gp_model = get_gp_model(ro_data = ro_data, x_map = model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)
        if args.continue_EI: 
            logger.info(f"Continue EI for a maximum of {args.n_EI} iterations") 
            x_star = ro_data.EI_infill["best_x"][-1]
            J_star_ref  = J_sampler.get_J_sample_ref(x_star, gp_model = gp_model, n_samp_J=args.n_samp_J, n_samp_gp=args.n_samp_gp)
        else:
            return gp_model, ro_data
    else: 
        ro_data.clear_EI_infill()
        opti_J = AcfnEM(gp_model, J_sampler, args)
        result = differential_evolution(opti_J.obj_fun_min_J, args = (args.n_samp_J_opt, args.n_samp_gp_opt), bounds = model.x_bounds, maxiter= args.EI_genopt_i, popsize = args.EI_genopt_pop, polish= False, constraints=model.constraints)
        x_star = torch.from_numpy(result.x)
        J_star_ref, _ = J_sampler.get_J_sample_moments(x=x_star, gp_model = gp_model, n_samp_J=args.n_samp_J, n_samp_gp=args.n_samp_gp)
        # x_star, _, _, J_star_ref = J_sampler.get_min_J(gp_model, model, ro_data.doe["xtilde"][:,:model.n_x], args.n_samp_gp, args.n_samp_J)
    ts = time.time()

    for i in range(args.n_EI):
        if i < args.n_EX: logger.info(f"EX Iteration {i+1}:")  
        else: logger.info(f"EI Iteration ({i+1}/{args.n_EI}):")  
        get_cuda_memory(logger, "EI - Cuda mem:")
        acfn = AcfnJEI(gp_model, J_star_ref, J_sampler, args, i)
        t0 = time.time()
        result = differential_evolution(acfn.obj_fun, args = (args.n_samp_J_opt, args.n_samp_gp_opt, False), bounds = model.x_bounds, maxiter= args.EI_genopt_i, popsize = args.EI_genopt_pop, polish= False, constraints=model.constraints)
        # logger.debug(f"     Avg. W2 distance of quantile distribution to normal dist = {torch.mean(torch.tensor(acfn.W2_dist)):.3f} and min/max dist = {torch.min(torch.tensor(acfn.W2_dist)):.3f} / {torch.max(torch.tensor(acfn.W2_dist)):.3f}")
        logger.debug(f"     Timing: EI Opt: t = {time.time() - t0:.3f} s - {acfn.obj_fun_calls} obj fun calls")
        logger.debug(f"     EI Opt: nit={result.nit}, nfev={result.nfev}, message={result.message}")
        x_c = torch.from_numpy(result.x)
        J_c = torch.mean(J_sampler.get_J_sample(x_c, gp_model, args.n_samp_J, args.n_samp_gp))
        ro_data.EI_infill["EI_val"].append(torch.from_numpy(np.array([-result.fun])))
        t1 = time.time()
        if args.plot_flag: add_EI_plot(ro_data, model, acfn, x_c, x_star, J_star_ref, args)
        logger.debug(f"     Timing: Gen Plot data: t = {time.time() - t1:.3f} s")

        # Stochstic domain infill
        for j in range(args.n_EI_se + args.n_EI_ufun):

            # Acquisition optimization
            acfn_uq = AcfnUQ(gp_model, x = x_c, model = model, J = J_c, measure = args.measure)
            z_star = opt_stochastic_domain_acfn(model = model, acfn_uq = acfn_uq, args = args, se_infill= (j+1) <= args.n_EI_se, logger = logger)
        
            # Observe
            tstart = time.time()
            xtilde = model.get_xtilde(x_c, z_star)
            new_y = model.eval_xtilde(xtilde)

            # Concatenate
            ro_data.append_EI_infill(xtilde, new_y)
            gp_model = get_gp_model(ro_data, model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)

            # Reevaluate
            J_c, J_c_std, J_c_ref = J_sampler.get_J_sample_moments(x=x_c, gp_model = gp_model, n_samp_J=args.n_samp_J, n_samp_gp=args.n_samp_gp, include_ref=True)
            logger.debug(f"     Timing: Observe, Retrain GP, Reevaluate: t = {time.time() - tstart:.3f} s")
            logger.debug(f"     J_c_std: {J_c_std:.3f}")
        
        # Reevaluate
        J_star, J_star_std, J_star_ref = J_sampler.get_J_sample_moments(x=x_star, gp_model = gp_model, n_samp_J=args.n_samp_J, n_samp_gp=args.n_samp_gp, include_ref=True)

        # Check min_quant
        if J_c_ref < J_star_ref:
            J_star_ref = J_c_ref
            J_star = J_c
            J_star_std = J_c_std
            x_star = x_c

        # Global opti
        opti_J = AcfnEM(gp_model, J_sampler, args)
        result = differential_evolution(opti_J.obj_fun_min_J, args = (args.n_samp_J_opt, args.n_samp_gp_opt), bounds = model.x_bounds, maxiter= args.EI_genopt_i, popsize = args.EI_genopt_pop, polish= False, constraints=model.constraints)
        x_glob = torch.from_numpy(result.x)
        J_glob, J_glob_std = J_sampler.get_J_sample_moments(x=x_glob, gp_model = gp_model, n_samp_J=args.n_samp_J, n_samp_gp=args.n_samp_gp)

        if J_glob > J_star:
            x_glob = x_star
            J_glob = J_star
            J_glob_std = J_star_std

        ro_data.EI_infill["n"] += 1
        ro_data.EI_infill["cand_x"].append(x_c)
        ro_data.EI_infill["cand_J"].append(J_c)
        ro_data.EI_infill["cand_J_std"].append(J_c_std)
        ro_data.EI_infill["best_x"].append(x_glob)
        ro_data.EI_infill["best_J"].append(J_glob)
        ro_data.EI_infill["best_J_std"].append(J_glob_std)
        ro_data.EI_infill["best_x_EI"].append(x_star)
        ro_data.EI_infill["best_J_EI"].append(J_star)
        ro_data.EI_infill["best_J_std_EI"].append(J_star_std)
        ro_data.EI_infill["model_calls"].append(args.n_EI_se + args.n_EI_ufun)

        log_infill(logger, model, x_c, J_c, x_star, J_star, x_glob, J_glob, ts, args)
        save_ro_data(ro_data=ro_data, model=model, gp_model=gp_model, args=args)
        if check_EI_conv(ro_data, logger, args): break
            
    return gp_model, ro_data

def perform_EM_infill(gp_model, model, J_sampler: RobustnessMeasure, ro_data, logger, args):
    if args.load_EM and not args.transfer_learning:
        if logger is not None:  logger.info(f"Load EM infill")
        ro_data.load_EM(args.sim_file)
        gp_model = get_gp_model(ro_data = ro_data, x_map = model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)
        if args.continue_EM: 
            logger.info(f"Continue EM for a maximum of {args.n_EM} iterations") 
            x_star, J_star, J_star_std = ro_data.get_best_sample(get_std = True)
        else:
            return gp_model, ro_data
    else:
        ro_data.clear_EM_infill()
        x_star, J_star, J_star_std = ro_data.get_best_sample(get_std = True)

    t0 = time.time()

    for i in range(args.n_EM):
        logger.info(f"EM Iteration ({i+1}/{args.n_EM}):")  
        get_cuda_memory(logger, "EM - Cuda mem:")
        # Global opti
        acfn = AcfnEM(gp_model, J_sampler, args)
        result = differential_evolution(acfn.obj_fun_min_J, args = (args.n_samp_J_opt, args.n_samp_gp_opt), bounds = model.x_bounds, maxiter= args.EM_genopt_i, popsize = args.EM_genopt_pop, polish= False, constraints=model.constraints)
        logger.debug(f"     EI Opt: nit={result.nit}, nfev={result.nfev}, message={result.message}")
        x_c = torch.from_numpy(result.x)
        J_c, J_c_std = J_sampler.get_J_sample_moments(x_c, gp_model, args.n_samp_J, args.n_samp_gp)
        if args.plot_flag: add_EM_plot(ro_data, model, gp_model, x_c, J_c, x_star, J_star, J_sampler, args, logger)
        
        # Stochstic domain infill
        for j in range(args.n_EM_se + args.n_EM_ufun):

            # Acquisition optimization
            acfn_uq = AcfnUQ(gp_model= gp_model, x = x_c, model = model, J = J_c, measure = args.measure)
            z_star = opt_stochastic_domain_acfn(model = model, acfn_uq = acfn_uq, args = args, se_infill= (j+1) <= args.n_EM_se, logger = logger)

            # Observe
            tstart = time.time()
            xtilde = model.get_xtilde(x_c, z_star)
            new_y = model.eval_xtilde(xtilde)

            # Concatenate
            ro_data.append_EM_infill(xtilde, new_y)
            gp_model = get_gp_model(ro_data, model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)

            # Reevaluate
            J_c, J_c_std = J_sampler.get_J_sample_moments(x_c, gp_model, args.n_samp_J, args.n_samp_gp)
            logger.debug(f"     Timing: Observe, Retrain GP, Reevaluate: t = {time.time() - tstart:.3f} s")
        
        # Reevaluate
        J_star, J_star_std = J_sampler.get_J_sample_moments(x_star, gp_model, args.n_samp_J, args.n_samp_gp)
        
        # Check min_quant
        if J_c < J_star:
            J_star = J_c
            J_star_std = J_c_std
            x_star = x_c

        ro_data.EM_infill["n"] += 1
        ro_data.EM_infill["cand_x"].append(x_c)
        ro_data.EM_infill["cand_J"].append(J_c)
        ro_data.EM_infill["cand_J_std"].append(J_c_std)
        ro_data.EM_infill["best_x"].append(x_star)
        ro_data.EM_infill["best_J"].append(J_star)
        ro_data.EM_infill["best_J_std"].append(J_star_std)
        ro_data.EM_infill["model_calls"].append(args.n_EM_se + args.n_EM_ufun)

        log_infill_EM(logger, model, x_c, J_c, x_star, J_star, t0, args)
        save_ro_data(ro_data=ro_data, model=model, gp_model=gp_model, args=args)
        if check_EM_conv(ro_data, logger, args): break

    return gp_model, ro_data

def perform_TR_infill(gp_model, model, J_sampler: RobustnessMeasure, ro_data, logger, args):
    if args.load_TR and not args.transfer_learning:
        if logger is not None:  logger.info(f"Load TR infill")
        ro_data.load_TR(args.sim_file)
        gp_model = get_gp_model(ro_data = ro_data, x_map = model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)
        if args.continue_TR: 
            logger.info(f"Continue TR for a maximum of {args.n_TR} iterations") 
            x_star, _ = ro_data.get_best_sample()
            gp_model_device = copy_gp_object(gp_model= gp_model, device = args.J_device, dtype = torch.float32)
            J_star = J_sampler.get_gp_mean_J(x = x_star, gp_model = gp_model_device, n_samp_J= args.n_samp_J_TR_opt)
            delta = ro_data.TR_infill["TR_delta"][-1]
        else:
            return gp_model, ro_data
    else:
        ro_data.clear_TR_infill()
        x_star, _ = ro_data.get_best_sample()
        gp_model_device = copy_gp_object(gp_model= gp_model, device = args.J_device, dtype = torch.float32)
        J_star = J_sampler.get_gp_mean_J(x = x_star, gp_model = gp_model_device, n_samp_J= args.n_samp_J_TR_opt)
        delta = None
    
    ts = time.time()
    for i in range(args.n_TR):
        logger.info(f"TR Iteration ({i+1}/{args.n_TR}):")  
        get_cuda_memory(logger, "TR - Cuda mem:")

        acfn = AcfnTR(gp_model = gp_model_device, model = model, x_star = x_star, J_sampler= J_sampler, delta = delta)
        t0 = time.time()
        result = differential_evolution(acfn.obj_fun_TR_numpy, args = (int(args.n_samp_J_TR_opt),int(args.n_rep_J_TR)), bounds = acfn.trust_reg.numpy().T, maxiter= args.TR_genopt_i, popsize = args.TR_genopt_pop, polish= True, constraints=model.constraints)

        logger.debug(f"     Timing: TR Opt: t = {time.time() - t0:.3f} s")
        x_c = torch.from_numpy(result.x)
        J_c = torch.from_numpy(np.array([result.fun]))[0]
        ro_data.TR_infill["cand_J_predict"].append(J_c)
        t1 = time.time()
        if args.plot_flag: add_TR_plot(ro_data, model, gp_model_device, x_c, J_c, x_star, J_star, acfn.trust_reg, J_sampler, args)

        del result, acfn 
        gc.collect()
        torch.cuda.empty_cache()
        
        logger.debug(f"     Timing: Gen Plot data: t = {time.time() - t1:.3f} s")

        # Stochstic domain infill
        for j in range(args.n_TR_se + args.n_TR_ufun):

            # Acquisition optimization
            acfn_uq = AcfnUQ(gp_model= gp_model, x = x_c, model = model, J = J_c, measure = args.measure)
            z_star = opt_stochastic_domain_acfn(model = model, acfn_uq = acfn_uq, args = args, se_infill= (j+1) <= args.n_TR_se, logger = logger)

            # Observe
            tstart = time.time()
            xtilde = model.get_xtilde(x_c, z_star)
            new_y = model.eval_xtilde(xtilde)

            # Concatenate
            ro_data.append_TR_infill(xtilde, new_y)
            gp_model = get_gp_model(ro_data, model.xtilde_map, add_noise = args.add_noise, max_attempts = args.max_attempts, logger = logger)

            # Reevaluate
            gp_model_device = copy_gp_object(gp_model= gp_model, device = args.J_device, dtype = torch.float32)
            J_c = J_sampler.get_gp_mean_J(x = x_c, gp_model = gp_model_device, n_samp_J= args.n_samp_J_TR_opt, n_rep = args.n_rep_J_TR)
            logger.debug(f"     Timing: Observe, Retrain GP, Reevaluate: t = {time.time() - tstart:.3f} s")
        # Reevaluate
        J_star = J_sampler.get_gp_mean_J(x = x_star, gp_model = gp_model_device, n_samp_J= args.n_samp_J_TR_opt, n_rep = args.n_rep_J_TR)

        # Compute Trust region --> x_star, J_star, x_c, J_c, J_c_predict
        delta = comp_TR_radius(x_star=x_star, J_star=J_star, x_c=x_c, J_c=J_c, J_c_predict=ro_data.TR_infill["cand_J_predict"][-1], model=model, independent_TR = False, delta_bounds=[0.01, 0.25], logger=logger)

        # Check min_quant
        if J_c < J_star:
            J_star = J_c
            x_star = x_c

        ro_data.TR_infill["n"] += 1
        ro_data.TR_infill["cand_x"].append(x_c)
        ro_data.TR_infill["cand_J"].append(J_c)
        ro_data.TR_infill["best_x"].append(x_star)
        ro_data.TR_infill["best_J"].append(J_star)
        ro_data.TR_infill["model_calls"].append(args.n_TR_se + args.n_TR_ufun)
        ro_data.TR_infill["TR_delta"].append(delta)

        log_infill_TR(logger, model, x_c, J_c, x_star, J_star, ts, args)
        save_ro_data(ro_data=ro_data, model=model, gp_model=gp_model, args=args)
        if check_TR_conv(ro_data, logger, args): break

    return gp_model, ro_data

# heuristics for default infill
def set_default_infill(model, logger, args):
    n_se = int(np.ceil(model.n_xi/3))
    n_ufun = int(np.ceil(model.n_xi/2)) 
    logger.info(f"Heuristics: \n")
    if args.n_doe == -1: 
        args.n_doe = model.n_dim * 5
        logger.info(f"    n_doe = {args.n_doe}")
    if args.n_doe_se == -1: 
        args.n_doe_se = 0 
        logger.info(f"    n_doe_se = {args.n_doe_se}")
    if args.n_doe_ufun == -1: 
        args.n_doe_ufun = 0
        logger.info(f"    n_doe_ufun = {args.n_doe_ufun}")
    if args.n_EI == -1: 
        args.n_EI = int(np.ceil(0.6*(args.n_doe)*4))
        logger.info(f"    n_EI = {args.n_EI}")
    if args.n_EI_se == -1: 
        args.n_EI_se = n_se
        logger.info(f"    n_   EI_se = {args.n_EI_se}")
    if args.n_EI_ufun == -1: 
        args.n_EI_ufun = n_ufun
        logger.info(f"    n_EI_ufun = {args.n_EI_ufun}")
    if args.n_EM == -1: 
        args.n_EM = int(np.ceil(0.2*(args.n_doe)*4))
        logger.info(f"    n_EM = {args.n_EM}")
    if args.n_EM_se == -1: 
        args.n_EM_se = n_se
        logger.info(f"    n_EM_se = {args.n_EM_se}")
    if args.n_EM_ufun == -1: 
        args.n_EM_ufun = n_ufun
        logger.info(f"    n_EM_ufun = {args.n_EM_ufun}")
    if args.n_TR == -1: 
        args.n_TR = int(np.ceil(0.2*(args.n_doe)*4))
        logger.info(f"    n_TR = {args.n_TR}")
    if args.n_TR_se == -1: 
        args.n_TR_se = n_se
        logger.info(f"    n_TR_se = {args.n_TR_se}")
    if args.n_TR_ufun == -1: 
        args.n_TR_ufun = n_ufun
        logger.info(f"    n_TR_ufun = {args.n_TR_ufun}")

    n_total = args.n_doe + args.n_doe * (args.n_doe_se + args.n_doe_ufun) + args.n_EI * (args.n_EI_se + args.n_EI_ufun) + args.n_EM * (args.n_EM_se + args.n_EM_ufun) + + args.n_TR * (args.n_TR_se + args.n_TR_ufun)
    logger.info(f"    Maximal number of model evals = {n_total}")

    return args
    

    

