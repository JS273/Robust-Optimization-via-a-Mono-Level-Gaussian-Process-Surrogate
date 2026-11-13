import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
sys.path.insert(0, parent_dir)

import torch, time, shutil
from ro.helper import get_gp_model, copy_gp_object
from models.library import get_model
from ro.acfn import RobustnessMeasure
import numpy as np
from utils.logger import init_logger
from utils.filemanager import save_script
from utils.datastructures import MonoLevelDatabase
import configargparse
from botorch.sampling.pathwise.posterior_samplers import draw_matheron_paths

torch.set_default_dtype(torch.float32)

def run(output_folder, tag, args, logger):

    # Input
    if args.dtype == "float":
        dtype = torch.float32
    elif args.dtype == "double":
        dtype = torch.float64

    # Get the model
    model_name = "Sabater1d"
    model = get_model(name = model_name)
    ro_data = MonoLevelDatabase()

    J_sampler = RobustnessMeasure(
                            measure=args.measure,
                            sampler=args.sampler,
                            tau = args.tau,
                            alpha = args.alpha,
                            J_device = args.J_device, 
                            model = model,
                            ref_stat = "mean",
                            logger = logger
                            )

    # Get the GP Model
    n_doe = 10
    sobol_engine = torch.quasirandom.SobolEngine(dimension=model.n_dim)
    init_train_u = sobol_engine.draw(n=n_doe).to(torch.float64)
    init_train_xtilde = model.xtilde_map.untransform(init_train_u)
    init_train_y = model.eval(init_train_xtilde[:,0], init_train_xtilde[:,1:3])

    logger.info(init_train_u.shape)
    logger.info(init_train_y.shape)

    logger.info(f"Input: {init_train_xtilde.dtype} {init_train_xtilde.device}")
    logger.info(f"Output:{init_train_y.dtype} {init_train_y.device}")

    ro_data.set_doe(init_train_xtilde, init_train_y, sobol_burn_in=0)
    gp_model = get_gp_model(ro_data, model.xtilde_map, add_noise = 1e-4)

    # Q sample size variation
    x_c = ro_data.doe["xtilde"][0,:model.n_x]
    gp_samples = args.fix_gp_samples
    J_sample_vec = np.linspace(50, args.max_samples, args.n_steps, dtype = int)
    t_vari_n_q = []

    logger.info(f"J sample variation \n")

    for n_samples_J in J_sample_vec:
        logger.info(f"Test points: {n_samples_J}")

        if args.sampler == "matheron":
            J_sampler.sampler = "matheron"
            gp_model_copy = copy_gp_object(gp_model, device = args.J_device, dtype= dtype)
            path = draw_matheron_paths(model=gp_model_copy, sample_shape=torch.Size([gp_samples]))
            start = time.time()
            J_glob, J_glob_std = J_sampler.get_J_sample_moments(x=x_c, gp_model = gp_model, n_samp_J=n_samples_J, n_samp_gp=gp_samples, matheron_path= path)
        elif args.sampler == "classical":
            J_sampler.sampler = "classical"
            start = time.time()
            J_glob, J_glob_std = J_sampler.get_J_sample_moments(x=x_c, gp_model = gp_model, n_samp_J=n_samples_J, n_samp_gp=gp_samples)

        t = time.time() - start
        logger.info(f"One Iter : {t}")
        t_vari_n_q.append(t)

        torch.cuda.empty_cache()

    # GP sample size variation
    logger.info(f"GP sample variation \n")

    x_c = ro_data.doe["xtilde"][0,:model.n_x]
    J_samples = args.fix_J_samples
    gp_sample_vec = np.linspace(10, args.max_samples, args.n_steps, dtype = int)
    t_vari_n_gp = []

    for gp_samples in gp_sample_vec:
        logger.info(f"GP Samples: {gp_samples}")
        if args.sampler == "matheron":
            J_sampler.sampler = "matheron"
            gp_model_copy = copy_gp_object(gp_model, device = args.J_device, dtype= dtype)
            path = draw_matheron_paths(model=gp_model_copy, sample_shape=torch.Size([gp_samples]))
            start = time.time()
            J_glob, J_glob_std = J_sampler.get_J_sample_moments(x=x_c, gp_model = gp_model, n_samp_J=J_samples, n_samp_gp=gp_samples)
        elif args.sampler == "classical":
            J_sampler.sampler = "classical"
            start = time.time()
            J_glob, J_glob_std = J_sampler.get_J_sample_moments(x=x_c, gp_model = gp_model, n_samp_J=J_samples, n_samp_gp=gp_samples)

        t = time.time() - start
        logger.info(f"One Iter : {t:.4} s")
        t_vari_n_gp.append(t)

        torch.cuda.empty_cache()

    filename = output_folder + f"/{tag}.npz" 
    np.savez(filename, 
             J_sample_vec = J_sample_vec, 
             t_vari_n_q = np.array(t_vari_n_q),
             gp_sample_vec = gp_sample_vec,
             t_vari_n_gp = np.array(t_vari_n_gp))


if __name__ == '__main__':

    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('--output_folder', default = None, type=str, help='Output_folder')
    p.add('--J_device', default = "cuda", type=str, help='device')
    p.add('--dtype', default = "float", type=str, help='data tyte')
    p.add("--measure", default="quantile", type=str)
    p.add('--sampler', default = "classical", type=str, help='sampling strategy')
    p.add("--tau", default = 0.8, type=float)
    p.add("--alpha", default = 0.5, type=float)
    p.add('--fix_gp_samples', default = 2000, type=int)
    p.add('--fix_J_samples', default = 2000, type=int)
    p.add('--n_steps', default = 2, type=int)
    p.add('--max_samples', default = 10000, type=float)

    args = p.parse_args()
    tag = f"{args.sampler}_{args.J_device}_{args.dtype}"

    if args.output_folder is None:
        output_folder = save_script(os.path.realpath(__file__), tag, max_res_folders= 7, max_daily_folders= 3)
    else:
        output_folder = args.output_folder
        os.makedirs(output_folder, exist_ok= True)
        target_file = output_folder + "/executed_file_" + tag + ".py"
        shutil.copy(os.path.realpath(__file__), target_file)

    logger = init_logger(output_folder, tag)
    logger.info("User input (cmd > config > default): \n" + p.format_values())

    run(output_folder, tag, args, logger)




