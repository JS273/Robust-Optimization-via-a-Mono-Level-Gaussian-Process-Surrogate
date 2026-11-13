"""High-level module for mono-level robust optimization.

This script configures and runs the MLRO workflow, including model
initialization, DOE generation, EI, EM, and trust-region infill phases.
It manages Gaussian-process updates, result storage, logging, and plotting
using command-line or YAML configuration options.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

import sys, os, json, configargparse
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, parent_dir)
import torch
torch.set_default_dtype(torch.float64)
import numpy as np
from models.library import get_model
from ro.opt_modules import create_init_gp, perform_doe_infill, perform_EI_infill, perform_EM_infill, perform_TR_infill, set_default_infill
from ro.acfn import RobustnessMeasure
from utils.datastructures import MonoLevelDatabase, save_ro_data
from utils.plotting import plot_best_evolution, plot_cand_evolution, plot_convergence
from utils.logger import init_logger
from utils.filemanager import create_result_folder


def main(args, logger):
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Init
    model_kwargs = json.loads(args.model_kwargs)
    model = get_model(name = args.model, **model_kwargs)
    args = set_default_infill(model, logger, args)
    ro_data = MonoLevelDatabase()
    J_sampler = RobustnessMeasure(
                                measure=args.measure,
                                sampler=args.sampler,
                                tau = args.tau,
                                alpha = args.alpha,
                                J_device = args.J_device, 
                                model = model,
                                ref_q= args.ref_q,
                                ref_stat = args.ref_stat,
                                verbose= args.verbose,
                                logger = logger
                                )

    # DOE Phase
    gp_model, ro_data, J_sampler = create_init_gp(model = model,
                                       J_sampler = J_sampler,
                                       ro_data = ro_data,
                                       logger = logger, 
                                       args = args)
    save_ro_data(ro_data=ro_data, model=model, gp_model=gp_model, args=args)

    # Infill DOE
    if args.n_doe_se + args.n_doe_ufun > 0:
        gp_model, ro_data = perform_doe_infill(gp_model = gp_model,
                                               model = model, 
                                               J_sampler = J_sampler,
                                               ro_data = ro_data,
                                               logger = logger,
                                               args = args)
        save_ro_data(ro_data=ro_data, model=model, gp_model=gp_model, args=args)

    # Infill EI
    if args.n_EI > 0:
        gp_model, ro_data = perform_EI_infill(gp_model = gp_model,
                                              model = model,
                                              J_sampler = J_sampler,
                                              ro_data = ro_data,
                                              logger = logger,
                                              args = args)

    # Infill EM
    if args.n_EM > 0:
        gp_model, ro_data = perform_EM_infill(gp_model = gp_model,
                                              model = model,
                                              J_sampler = J_sampler,
                                              ro_data = ro_data,
                                              logger = logger,
                                              args = args)

    # Infill TR
    if args.n_TR > 0:
        gp_model, ro_data = perform_TR_infill(gp_model = gp_model,
                                              model = model,
                                              J_sampler = J_sampler,
                                              ro_data = ro_data,
                                              logger = logger,
                                              args = args)

    # Plotting 
    plot_best_evolution(ro_data=ro_data, model = model, args = args, output_folder = output_folder, tag = args.tag)
    plot_cand_evolution(ro_data=ro_data, model = model, args = args, output_folder = output_folder, tag = args.tag)
    plot_convergence(ro_data=ro_data, model = model, args = args, output_folder=output_folder, tag=args.tag)

if __name__ == "__main__":
    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('--config', is_config_file=True, help='Path to the config file')
    # RO DOE
    parser.add("--n_doe", default=15, type=int)
    parser.add("--n_doe_se", default=1, type=int)
    parser.add("--n_doe_ufun", default=0, type=int)
    parser.add("--n_EI", default=10, type=int)
    parser.add("--n_EX", default=0, type=int)
    parser.add("--n_EI_se", default=1, type=int)
    parser.add("--n_EI_ufun", default=2, type=int)
    parser.add("--n_EM", default=10, type=int)
    parser.add("--n_EM_se", default=1, type=int)
    parser.add("--n_EM_ufun", default=2, type=int)
    parser.add("--n_TR", default=10, type=int)
    parser.add("--n_TR_se", default=0, type=int)
    parser.add("--n_TR_ufun", default=2, type=int)
    # Load sim
    parser.add("--sim_file", default=None, type=str)
    parser.add("--transfer_learning", action='store_true', default=False)
    parser.add("--load_doe", action='store_true', default=False)
    parser.add("--load_doe_infill", action='store_true', default=False)
    parser.add("--load_EI", action='store_true', default=False)
    parser.add("--load_EM", action='store_true', default=False)
    parser.add("--load_TR", action='store_true', default=False)
    parser.add("--continue_doe", action='store_true', default=False)
    parser.add("--continue_doe_infill", action='store_true', default=False)
    parser.add("--catch_up_doe_infill", action='store_true', default=False)
    parser.add("--continue_EI", action='store_true', default=False)
    parser.add("--continue_EM", action='store_true', default=False)
    parser.add("--continue_TR", action='store_true', default=False)
    # Convergence
    parser.add("--conv_EI", default = '{"switch": 0, "eps": 0.05, "n_ave": 2}')
    parser.add("--conv_x", default = '{"switch": 0, "abs_x": 1e-5, "rel_x": 1e-2, "n_ave": 2}')
    parser.add("--conv_y", default = '{"switch": 0, "abs_y": 1e-4, "rel_y": 1e-2, "n_ave": 2}')
    # Optimizer 
    parser.add("--EI_genopt_i", default=15, type=int)
    parser.add("--EI_genopt_pop", default=10, type=int)
    parser.add("--EM_genopt_i", default=15, type=int)
    parser.add("--EM_genopt_pop", default=10, type=int)
    parser.add("--TR_genopt_i", default=15, type=int)
    parser.add("--TR_genopt_pop", default=10, type=int)
    parser.add("--UQ_genopt_i", default=15, type=int)
    parser.add("--UQ_genopt_pop", default=10, type=int)
    # GP Hyperparameter tuning
    parser.add("--max_attempts", default=5, type=int)
    # Quantile Sampling
    parser.add("--measure", default="quantile", type=str)
    parser.add("--tau", default = 0.8, type=float)
    parser.add("--alpha", default = 0.5, type=float)
    parser.add("--sampler", default="matheron", type=str)
    parser.add("--J_device", default="cuda", type=str)
    parser.add("--n_samp_gp", default=5000, type=int)
    parser.add("--n_samp_J", default=5000, type=int)
    parser.add("--n_samp_gp_opt", default=2000, type=int)
    parser.add("--n_samp_J_opt", default=2000, type=int)
    parser.add("--n_samp_J_TR_opt", default=1e5, type=int)
    parser.add("--n_rep_J_TR", default=1, type=int)
    parser.add("--plot_n_samp_J", default = 2000, type=int)
    parser.add("--plot_n_samp_gp", default = 2000, type=int)
    # General 
    parser.add("--doe_burn_in", default=0, type = int)
    parser.add("--ref_stat", default="mean", type = str)
    parser.add("--ref_q", default=0.9, type = float)
    parser.add("--EI_type", default="analytical", type = str)
    parser.add("--seed", default=45, type=int)
    parser.add("--output_folder", default=None, type=str)
    parser.add("--tag", default="ro", type=str)
    parser.add("--verbose", action='store_true', default=False)
    parser.add("--log_level", default=1, type=int)
    # Model
    parser.add("--model", default = "Sabater2d", type=str)
    parser.add("--model_kwargs", default = '{"uq_domain": "linear"}')
    parser.add("--add_noise", default = 1e-4, type=float)
    parser.add("--plot_flag", action='store_true', default=False)
    parser.add("--plot_n_points", default = 20, type=int)
    parser.add("--plot_n_rep", default=2, type=int)
    args = parser.parse_args()

    if args.output_folder is None:
        tag = f"{args.model}_ndoe_{args.n_doe}_n_EI_{args.n_EI}"
        output_folder = create_result_folder(tag, max_res_folders = 4, max_daily_folders= 4)
    else:
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok= True)

    # Create output folder
    logger = init_logger(output_folder, f"{args.tag}_log", log_level= args.log_level)
    logger.info("User input (cmd > config > default): \n" + parser.format_values())
    args.output_folder = output_folder
    try:
        main(args, logger)
    except Exception:
        # 2) this logs the full stack trace automatically
        logger.exception("Unhandled exception in main()")
        # if you want the process to exit with a nonzero code:
        sys.exit(1)

    
    