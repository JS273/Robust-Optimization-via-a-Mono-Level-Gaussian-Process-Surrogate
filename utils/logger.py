"""Logging utilities for robust optimization workflows.

This module provides functions for initializing file and console loggers,
recording design-of-experiments and infill progress, reporting optimization
results and timings, and monitoring CUDA memory usage during computations.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

import logging, time, torch
import subprocess

def init_logger(save_dir, logfile, log_level= 1, verbose = True):

    log_file = save_dir + "/" + logfile + ".log"

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    if log_level == 1:
        file_handler.setLevel(logging.INFO)
    elif log_level == 2:
        file_handler.setLevel(logging.DEBUG)
        
    logger.addHandler(file_handler)

    # Stream handler to log messages to the console
    if verbose:
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter.default_time_format = '%H:%M:%S'
        console_handler.setFormatter(console_formatter)
        if log_level == 1:
            console_handler.setLevel(logging.INFO)
        elif log_level == 2:
            console_handler.setLevel(logging.DEBUG)
            
        logger.addHandler(console_handler)

    commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()

    logger.info(f"-------------------------------")
    logger.info(f"--------Start logging ---------")
    logger.info(f"-------------------------------\n")
    logger.info(f"Current git hash: {commit_hash}")

    return logger

def log_doe(logger, model, ro_data, init_train_y, init_train_xtilde, args):
    cand_x, cand_J= ro_data.doe["cand_x"], ro_data.doe["cand_J"]
    logger.info(f"Create Initial GP")
    init_train_y = ro_data.doe["y"]
    init_train_xtilde = ro_data.doe["xtilde"]
    logger.info(f"    Output samples : {init_train_y.shape}")
    logger.info(f"    Input  samples : {init_train_xtilde.shape}")
    formatted_x_cand = [f"{x:.2f}" for x in torch.atleast_1d(cand_x)]
    logger.info(f"    Candidate x = {formatted_x_cand} with q = {cand_J:.2f}")
    try:
        opt_x, opt_q = model.load_opt(measure = args.measure, tau = args.tau, alpha = args.alpha)
        formatted_x_opt = [f"{x:.2f}" for x in torch.atleast_1d(opt_x)]
        logger.info(f"    Optimum   x = {formatted_x_opt} with q = {opt_q[0]:.2f} \n")
    except:
        logger.info(f"    No ref. optimum available \n")

def log_infill(logger, model, x_c, J_c, x_star, J_star, x_glob, J_glob, t0, args):
    formatted_x_cand = [f"{x:.2f}" for x in torch.atleast_1d(x_c)]
    formatted_x_star = [f"{x:.2f}" for x in torch.atleast_1d(x_star)]
    formatted_x_glob = [f"{x:.2f}" for x in torch.atleast_1d(x_glob)]
    logger.info(f"    Candidate:    x = {formatted_x_cand} with q = {J_c:.2f}")
    logger.info(f"    Best in Samp: x = {formatted_x_star} with q = {J_star:.2f}")
    logger.info(f"    Best in Glob: x = {formatted_x_glob} with q = {J_glob:.2f}")
    try:
        opt_x, opt_q = model.load_opt(measure = args.measure, tau = args.tau, alpha = args.alpha)
        formatted_x_opt = [f"{x:.2f}" for x in torch.atleast_1d(opt_x)]
        logger.info(f"    Optimum:      x = {formatted_x_opt} with q = {opt_q[0]:.2f} \n")
    except:
        logger.info(f"    No ref. optimum available \n")
    logger.debug(f"   Timing: Total EI iter {time.time() - t0:.3f} s\n")

def log_infill_TR(logger, model, x_c, J_c, x_star, J_star, t0, args):
    formatted_x_cand = [f"{x:.2f}" for x in torch.atleast_1d(x_c)]
    formatted_x_star = [f"{x:.2f}" for x in torch.atleast_1d(x_star)]
    logger.info(f"    Candidate:    x = {formatted_x_cand} with q = {J_c:.2f}")
    logger.info(f"    Best:         x = {formatted_x_star} with q = {J_star:.2f}")
    try:
        opt_x, opt_q = model.load_opt(measure = args.measure, tau = args.tau, alpha = args.alpha)
        formatted_x_opt = [f"{x:.2f}" for x in torch.atleast_1d(opt_x)]
        logger.info(f"    Optimum:      x = {formatted_x_opt} with q = {opt_q[0]:.2f} \n")
    except:
        logger.info(f"    No ref. optimum available \n")
    logger.debug(f"        Timing: Total EI iter {time.time() - t0:.3f} s\n")

def log_infill_EM(logger, model, x_c, J_c, x_star, J_star, t0, args):
    formatted_x_cand = [f"{x:.2f}" for x in torch.atleast_1d(x_c)]
    formatted_x_star = [f"{x:.2f}" for x in torch.atleast_1d(x_star)]
    logger.info(f"    Candidate:    x = {formatted_x_cand} with q = {J_c:.2f}")
    logger.info(f"    Best:         x = {formatted_x_star} with q = {J_star:.2f}")
    try:
        opt_x, opt_q = model.load_opt(measure = args.measure, tau = args.tau, alpha = args.alpha)
        formatted_x_opt = [f"{x:.2f}" for x in torch.atleast_1d(opt_x)]
        logger.info(f"    Optimum:      x = {formatted_x_opt} with q = {opt_q[0]:.2f} \n")
    except:
        logger.info(f"    No ref. optimum available \n")
    logger.debug(f"        Timing: Total EI iter {time.time() - t0:.3f} s\n")

def get_cuda_memory(logger, tag):
    # total bytes currently allocated by tensors on GPU 0
    logger.debug(tag)
    alloc_bytes = torch.cuda.memory_allocated(device=0)

    # total bytes reserved (cached) by the CUDA allocator on GPU 0
    reserved_bytes = torch.cuda.memory_reserved(device=0)

    logger.debug(f"Allocated: {alloc_bytes/1024**2:.1f} MB")
    logger.debug(f"Reserved : {reserved_bytes/1024**2:.1f} MB")

    # If you want a full human-readable summary:
    # logger.info(torch.cuda.memory_summary(device=0, abbreviated=False))