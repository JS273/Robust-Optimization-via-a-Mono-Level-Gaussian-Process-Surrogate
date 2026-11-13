"""High-level module for computing robust optimization reference solutions.

This script initializes a selected model, configures the reference
optimization problem, and computes a robust optimum using the specified
measure, sampling method, and optimization settings. It manages random seeds,
logging, output directories, and persistence of the resulting reference data.

Model and optimization settings can be provided through command-line
arguments or a YAML configuration file.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

import sys
import os
import json
import numpy as np
import torch
import configargparse

parent_dir = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
)
sys.path.insert(0, parent_dir)

from utils.logger import init_logger
from utils.filemanager import create_result_folder
from models.library import get_model
from models.reference import ReferenceSpec, ReferenceOpt

torch.set_default_dtype(torch.float64)


def parse_json_kwargs(kwargs_str, arg_name="kwargs"):
    if kwargs_str is None:
        return {}
    if isinstance(kwargs_str, dict):
        return kwargs_str
    if isinstance(kwargs_str, str):
        kwargs_str = kwargs_str.strip()
        if kwargs_str == "":
            return {}
        try:
            return json.loads(kwargs_str)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Could not parse {arg_name} as JSON string: {kwargs_str}"
            ) from exc
    raise ValueError(f"Unsupported type for {arg_name}: {type(kwargs_str)}")

def format_tau(tau):
    return str(tau).replace(".", "p")


def format_n_samples(n):
    s = f"{n:.0e}"
    mantissa, exp = s.split("e")
    exp = str(int(exp))
    mantissa = mantissa.rstrip("0").rstrip(".")
    mantissa = mantissa.replace(".", "p")
    return f"{mantissa}e{exp}"

def main(args, spec, logger):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_kwargs = parse_json_kwargs(args.model_kwargs, arg_name="model_kwargs")
    model = get_model(name=args.model, **model_kwargs)

    logger.info("Model metadata:")
    for k, v in model.metadata.items():
        logger.info(f"  {k}: {v}")

    logger.info("Reference spec:")
    for k, v in spec.to_dict().items():
        logger.info(f"  {k}: {v}")

    optimization = ReferenceOpt(
        root=args.output_folder,
        device=args.device,
        logger=logger,
    )

    if args.measure == "mean_std": normalization = model.get_mean_std_normalization()
    else: normalization = None

    ref_result = optimization.run(model, spec, normalization=normalization)
    optimization.save(model, spec, ref_result)

    logger.info("Finished run.")

if __name__ == "__main__":
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser.add("--seed", default=45, type=int)
    parser.add("--device", default="cpu", type=str)
    parser.add("--output_folder", default=None, type=str)
    parser.add("--log_level", default=1, type=int)

    parser.add("--model", default="Sabater1d", type=str)
    parser.add("--model_kwargs", default="{}", type=str)

    parser.add("--measure", default="quantile", type=str)
    parser.add("--alpha", default=0.5, type=float)
    parser.add("--tau", default=0.95, type=float)
    parser.add("--n_samples", default=1000000, type=int)
    parser.add("--n_points", default=100, type=int)
    parser.add("--popsize", default=25, type=int)
    parser.add("--tol", default=0.0005, type=float)
    parser.add("--maxiter", default=100, type=int)
    parser.add("--method", default="mc", type=str)

    args = parser.parse_args()

    model_kwargs = parse_json_kwargs(args.model_kwargs, arg_name="model_kwargs")
    model = get_model(name=args.model, **model_kwargs)

    spec = ReferenceSpec(
        measure=args.measure,
        alpha=args.alpha,
        tau=args.tau,
        n_samples=args.n_samples,
        n_points=args.n_points,
        popsize=args.popsize,
        tol=args.tol,
        maxiter=args.maxiter,
        seed=args.seed,
        method=args.method,
    )

    if args.output_folder is None:
        tag = f"{model.name}_tau{str(args.tau).replace('.', 'p')}"
        output_folder = create_result_folder(tag, max_daily_folders=4, max_res_folders=4)
    else:
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok=True)

    if args.measure == "quantile":
        log_name = f"{model.name}_tau{format_tau(args.tau)}"
    elif args.measure == "mean_std":
        log_name = f"{model.name}_alpha{format_tau(args.alpha)}"

    logger = init_logger(output_folder, log_name, log_level=args.log_level)
    logger.info("User input (cmd > config > default): \n" + parser.format_values())

    args.output_folder = output_folder

    try:
        main(args, spec, logger)
    except Exception:
        logger.exception("Unhandled exception in main()")
        sys.exit(1)
