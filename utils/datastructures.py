"""Data-management for robust optimization.

This module provides functions for saving and loading optimization data,
database classes for storing DOE and infill results.
Moreover it contains utilities for creating, analyzing, and interpolating one-, two-, and three-dimensional
parameter grids.

Author
------
Julius Schultz

Date
----
2026-10-16

"""

import numpy as np
from scipy import interpolate
import pickle
import torch

def save_ro_data(ro_data, model, gp_model, args):
    filename = f'{args.tag}_data'
    with open(f'{args.output_folder}/{filename}.pkl', 'wb') as f:
        pickle.dump(ro_data, f)
        try:
            pickle.dump(model, f)
        except:
            pickle.dump(None, f)
            print("model could not be pickled")
        pickle.dump(gp_model, f)
        pickle.dump(args, f)

def load_ro_data(data_file):
    if data_file is None:
        raise ValueError("Data file is None. Please specify a file to load data")
    loaded_objects = []
    with open(f'{data_file}', 'rb') as f:
        try:
            while True:
                loaded_objects.append(pickle.load(f))
        except EOFError:
            pass

    ro_data, _, _, _ = loaded_objects
    return ro_data

class MonoLevelDatabase:
    def __init__(self):
        self.doe =        {"xtilde": torch.tensor([]), "y": torch.tensor([]), "n": 0, "sobol_burn_in": [], "cand_x": [], "cand_J": [], "cand_J_std": [], "model_calls": [], "plot_data": []}
        self.doe_infill = {"xtilde": torch.tensor([]), "y": torch.tensor([]), "n": 0, "plot_data": []}
        self.EI_infill =  {"xtilde": torch.tensor([]), "y": torch.tensor([]), "n": 0, "cand_x": [], "cand_J": [], "cand_J_std": [],"best_x": [], "best_J": [], "best_J_std": [], "best_x_EI": [], "best_J_EI": [], "best_J_std_EI": [], "model_calls": [], "plot_data": [], "EI_val": []}
        self.EM_infill =  {"xtilde": torch.tensor([]), "y": torch.tensor([]), "n": 0, "cand_x": [], "cand_J": [],"best_x": [], "best_J": [], "best_J_std": [],"cand_J_predict": [], "model_calls": [], "plot_data": []}
        self.TR_infill =  {"xtilde": torch.tensor([]), "y": torch.tensor([]), "n": 0, "cand_x": [], "cand_J": [],"best_x": [], "best_J": [],"cand_J_predict": [], "model_calls": [], "delta_x_abs": [], "delta_x_rel": [], "TR_delta": [], "plot_data": []}

    def set_doe(self, xtilde, y, sobol_burn_in):
        self.doe["xtilde"] = xtilde
        self.doe["y"] = y
        self.doe["n"] = xtilde.shape[0]
        self.doe["sobol_burn_in"] = sobol_burn_in

    def append_doe(self, xtilde, y):
        self.doe["xtilde"] = torch.concatenate((self.doe["xtilde"], xtilde), axis = 0)
        self.doe["y"] = torch.concatenate((self.doe["y"], y), axis = 0)
        self.doe["n"] = self.doe["n"] + xtilde.shape[0]

    def append_doe_infill(self, xtilde, y):
        self.doe_infill["xtilde"] = torch.concatenate((self.doe_infill["xtilde"], xtilde.unsqueeze(0)), axis = 0)
        self.doe_infill["y"] = torch.concatenate((self.doe_infill["y"], y), axis = 0)

    def append_EI_infill(self, xtilde, y):
        self.EI_infill["xtilde"] = torch.concatenate((self.EI_infill["xtilde"], xtilde.unsqueeze(0)), axis = 0)
        self.EI_infill["y"] = torch.concatenate((self.EI_infill["y"], y), axis = 0)

    def append_EM_infill(self, xtilde, y):
        self.EM_infill["xtilde"] = torch.concatenate((self.EM_infill["xtilde"], xtilde.unsqueeze(0)), axis = 0)
        self.EM_infill["y"] = torch.concatenate((self.EM_infill["y"], y), axis = 0)

    def append_TR_infill(self, xtilde, y):
        self.TR_infill["xtilde"] = torch.concatenate((self.TR_infill["xtilde"], xtilde.unsqueeze(0)), axis = 0)
        self.TR_infill["y"] = torch.concatenate((self.TR_infill["y"], y), axis = 0)

    def clear_doe_infill(self):
        self.doe_infill["xtilde"] = torch.tensor([])
        self.doe_infill["y"] = torch.tensor([])
        self.doe_infill["plot_data"] = []

    def clear_EI_infill(self):
        self.EI_infill["xtilde"] = torch.tensor([])
        self.EI_infill["y"] = torch.tensor([])
        self.EI_infill["plot_data"] = []
        self.EI_infill["n"] = 0
        self.EI_infill["cand_x"] = []
        self.EI_infill["cand_J"] = []
        self.EI_infill["cand_J_std"] = []
        self.EI_infill["best_x"] = []
        self.EI_infill["best_J"] = []
        self.EI_infill["best_J_std"] = []

    def clear_EM_infill(self):
        self.EM_infill["xtilde"] = torch.tensor([])
        self.EM_infill["y"] = torch.tensor([])
        self.EM_infill["plot_data"] = []
        self.EM_infill["n"] = 0
        self.EM_infill["cand_x"] = []
        self.EM_infill["cand_J"] = []
        self.EM_infill["cand_J_std"] = []
        self.EM_infill["best_x"] = []
        self.EM_infill["best_J"] = []
        self.EM_infill["best_J_std"] = []

    def clear_TR_infill(self):
        self.TR_infill["xtilde"] = torch.tensor([])
        self.TR_infill["y"] = torch.tensor([])
        self.TR_infill["plot_data"] = []
        self.TR_infill["n"] = 0

    def reuse_sim(self, args, logger):
        ro_data = load_ro_data(args.sim_file)
        n = ro_data.doe["n"]
        logger.info(f"      Load doe")
        if args.load_doe_infill:
            logger.info(f"      Load doe infill")
            ro_data.doe["xtilde"] = torch.concatenate((ro_data.doe["xtilde"],
                                                       ro_data.doe_infill["xtilde"]), axis = 0)
            ro_data.doe["y"] = torch.concatenate((ro_data.doe["y"],
                                                  ro_data.doe_infill["y"]), axis = 0)
            n += ro_data.doe_infill["xtilde"].shape[0]

        if args.load_EI:
            logger.info(f"      Load EI infill")
            ro_data.doe["xtilde"] = torch.concatenate((ro_data.doe["xtilde"],
                                                       ro_data.EI_infill["xtilde"]), axis = 0)
            ro_data.doe["y"] = torch.concatenate((ro_data.doe["y"],
                                                  ro_data.EI_infill["y"]), axis = 0)
            n += ro_data.EI_infill["xtilde"].shape[0]

        if args.load_EM:
            logger.info(f"      Load EM infill")
            ro_data.doe["xtilde"] = torch.concatenate((ro_data.doe["xtilde"],
                                                       ro_data.EM_infill["xtilde"]), axis = 0)
            ro_data.doe["y"] = torch.concatenate((ro_data.doe["y"],
                                                  ro_data.EM_infill["y"]), axis = 0)
            n += ro_data.EM_infill["xtilde"].shape[0]

        if args.load_TR:
            logger.info(f"      Load TR infill")
            ro_data.doe["xtilde"] = torch.concatenate((ro_data.doe["xtilde"],
                                                        ro_data.TR_infill["xtilde"]), axis = 0)
            ro_data.doe["y"] = torch.concatenate((ro_data.doe["y"],
                                                  ro_data.TR_infill["y"]), axis = 0)
            n += ro_data.TR_infill["xtilde"].shape[0]

        logger.info(f"      Load {n} samples to DOE")
        ro_data.doe["n"] = n
        self.doe = ro_data.doe

    def load_doe(self, doe_file):
        ro_data = load_ro_data(doe_file)
        n_doe = ro_data.doe["n"]
        self.doe = ro_data.doe
        return n_doe, ro_data.doe["sobol_burn_in"]

    def load_doe_infill(self, doe_file):
        ro_data = load_ro_data(doe_file)
        n_doe_infill = ro_data.doe_infill["n"]
        self.doe_infill = ro_data.doe_infill
        return n_doe_infill

    def load_EI(self, doe_file):
        ro_data = load_ro_data(doe_file)
        self.EI_infill = ro_data.EI_infill

    def load_EM(self, doe_file):
        ro_data = load_ro_data(doe_file)
        self.EM_infill = ro_data.EM_infill

    def load_TR(self, doe_file):
        ro_data = load_ro_data(doe_file)
        self.TR_infill = ro_data.TR_infill

    def append_doe_plot(self, plt_data):
        self.doe["plot_data"].append(plt_data)

    def append_doe_infill_plot(self, plt_data):
        self.doe_infill["plot_data"].append(plt_data)

    def append_EI_plot(self, plt_data):
        self.EI_infill["plot_data"].append(plt_data)

    def append_EM_plot(self, plt_data):
        self.EM_infill["plot_data"].append(plt_data)

    def append_TR_plot(self, plt_data):
        self.TR_infill["plot_data"].append(plt_data)
        
    def get_all_samples(self):
        xtilde = torch.concatenate((self.doe["xtilde"],
                                    self.doe_infill["xtilde"],
                                    self.EI_infill["xtilde"],
                                    self.EM_infill["xtilde"],
                                    self.TR_infill["xtilde"]), axis = 0)

        y = torch.concatenate((self.doe["y"],
                               self.doe_infill["y"],
                               self.EI_infill["y"],
                               self.EM_infill["y"],
                               self.TR_infill["y"]), axis = 0)
        
        return xtilde, y
        
    def get_best_sample(self, get_std = False):
        x_best = torch.stack(self.EI_infill["best_x"] + self.EM_infill["best_x"] + self.TR_infill["best_x"])
        J_best = torch.stack(self.EI_infill["best_J"] + self.EM_infill["best_J"] +  self.TR_infill["best_J"])        
        J_best_std = torch.stack(self.EI_infill["best_J_std"] + self.EM_infill["best_J_std"])        

        idx = torch.argmin(J_best)

        if get_std:
            return x_best[idx], J_best[idx], J_best_std[idx]
        else:
            return x_best[idx], J_best[idx]
        
class UQDatabase:
    def __init__(self, device = "cpu"):
        self.device = device
        self.doe =        {"z": torch.tensor([], device= device), "y": torch.tensor([], device= device), "n": 0, "sobol_burn_in": []}
        self.UQ_infill =  {"z": torch.tensor([], device= device), "y": torch.tensor([], device= device), "n": 0}

    def set_doe(self, z, y, sobol_burn_in):
        self.doe["z"] = z
        self.doe["y"] = y
        self.doe["n"] = z.shape[0]
        self.doe["sobol_burn_in"] = sobol_burn_in

    def append_UQ_infill(self, z, y):
        self.UQ_infill["z"] = torch.concatenate((self.UQ_infill["z"], z.unsqueeze(0)), axis = 0)
        self.UQ_infill["y"] = torch.concatenate((self.UQ_infill["y"], y), axis = 0)

    def get_all_samples(self):
        z = torch.concatenate((self.doe["z"],
                                    self.UQ_infill["z"]), axis = 0)

        y = torch.concatenate((self.doe["y"],
                               self.UQ_infill["y"]), axis = 0)
        
        return z, y


class Grid:
    def __init__(self, *dimension_samples, labels = ["x", "y", "z"]):

        self.dimension_samples = dimension_samples
        self.labels = labels

        self.n_dim = len(dimension_samples)
        self.n_nodes_per_dim = []
        self.min_per_dim = np.array([])
        self.max_per_dim = np.array([])
        self.l_per_dim = np.array([])
        self.dx_per_dim = np.array([])

        for sample in dimension_samples:
            self.n_nodes_per_dim.append(len(sample))
            self.min_per_dim = np.append(self.min_per_dim, np.min(sample))
            self.max_per_dim = np.append(self.max_per_dim, np.max(sample))
            self.l_per_dim = np.append(self.l_per_dim, np.max(sample) - np.min(sample))
            if sample.size > 1:
                self.dx_per_dim = np.append(self.dx_per_dim, sample[1] - sample[0])

        self.nodes, self.meshgrid_mat = self.full_factorial_grid()
        self.size = self.nodes.shape[0]
    
    @classmethod
    def create_2D_grid_from_coord(cls,x,y):
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        flag_x_nodes_greater_0 = x > 1e-15
        flag_y_nodes_greater_0 = y > 1e-15
        delta_x = np.min(x[flag_x_nodes_greater_0])
        delta_y = np.min(y[flag_y_nodes_greater_0])

        n_elem_x = int(x_max / delta_x)
        n_elem_y = int(y_max / delta_y)

        x_discretization = np.linspace(x_min, x_max, n_elem_x + 1)
        y_discretization = np.linspace(y_min, y_max, n_elem_y + 1)
        grid = cls(x_discretization,y_discretization)

        return grid
    
    @classmethod
    def create_3D_grid_from_coord(cls,x,y,z):
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        z_min = np.min(z)
        z_max = np.max(z)

        flag_x_nodes_greater_0 = x > 1e-15
        flag_y_nodes_greater_0 = y > 1e-15
        flag_z_nodes_greater_0 = z > 1e-15
        delta_x = np.min(x[flag_x_nodes_greater_0])
        delta_y = np.min(y[flag_y_nodes_greater_0])
        delta_z = np.min(z[flag_z_nodes_greater_0])

        n_elem_x = int(x_max / delta_x)
        n_elem_y = int(y_max / delta_y)
        n_elem_z = int(z_max / delta_z)

        x_discretization = np.linspace(x_min, x_max, n_elem_x + 1)
        y_discretization = np.linspace(y_min, y_max, n_elem_y + 1)
        z_discretization = np.linspace(z_min, z_max, n_elem_z + 1)
        grid = cls(x_discretization,y_discretization,z_discretization)

        return grid

    def full_factorial_grid(self):

        n_factorial = 1
        for sample in self.dimension_samples:
            n_factorial = n_factorial * len(sample)

        nodes = np.zeros((n_factorial, self.n_dim))
        meshgrid_mat = np.meshgrid(*self.dimension_samples)

        for idx, parameter in enumerate(meshgrid_mat):
            nodes[:,idx] = parameter.flatten()

        return nodes, meshgrid_mat
    
    def get_oat_sample_mat(self):
        mean_val = []
        for dim_sample in self.dimension_samples:
            mean_val.append(np.mean(dim_sample))

        mean_val = np.array(mean_val)

        sample_mat = []

        for idx, dim_sample in enumerate(self.dimension_samples):
            mean_mat = np.tile(mean_val, (len(dim_sample), 1))
            mean_mat[:,idx] = dim_sample
            sample_mat.append(mean_mat)


        sample_mat = np.vstack(sample_mat)

        return sample_mat

    def reshape_data(self, fun_evals):
        
        fun_evals_grid = []

        if self.n_dim == 2:
            fun_evals_grid = fun_evals.reshape(self.n_nodes_per_dim[1], self.n_nodes_per_dim[0])

        return fun_evals_grid

    def create_interpolator(self, fun_evals):

        interpolator = []

        if self.n_dim == 1:
            interpolator = interpolate.interp1d(self.nodes[:,0], fun_evals, axis = 0)
        elif self.n_dim == 2:
            zz = self.reshape_data(fun_evals = fun_evals)
            interpolator = interpolate.RectBivariateSpline(self.dimension_samples[0], self.dimension_samples[1], zz.T)

        return interpolator
    
    def create_legend(self, dim = 0):

        legendText = []

        for i in range(self.n_nodes_per_dim[dim]):
            legendText.append(f'{self.labels[dim]} = {self.dimension_samples[dim][i] :.2}')

        return legendText

    def sample_picker(self, *condition):
        
        a = []

        return a
    
    def get_euclid_distance(self, node_ids, l_max):
        
        distance_mat = np.ones((self.size, len(node_ids))) * 10000

        for idx, node_id in enumerate(node_ids):
            node_x = self.nodes[node_id, 0]
            node_y = self.nodes[node_id, 1]

            nodes_in_square = self.get_nodes_in_square(node_id, l_max)

            x_coords = self.nodes[nodes_in_square,0]
            y_coords = self.nodes[nodes_in_square,1]

            delta_x = x_coords - node_x
            delta_y = y_coords - node_y

            distance_mat[nodes_in_square,idx] = np.sqrt(np.square(delta_x) + np.square(delta_y))

        return distance_mat

    def get_nodes_in_square(self, node_id, l):
        
        n_nodes_x = l/self.dx_per_dim[0]
        n_idx_x = int(np.floor(n_nodes_x/2))

        n_nodes_y = l/self.dx_per_dim[1]
        n_idx_y = int(np.floor(n_nodes_y/2))

        node_ids = np.arange(0,self.size, dtype= int)

        node_ids_mat = self.reshape_data(node_ids)

        node_id_in_mat = np.argwhere(node_ids_mat == node_id)
        x_id = node_id_in_mat[0,0]
        y_id = node_id_in_mat[0,1]

        node_ids_in_square = node_ids_mat[np.max((x_id-n_idx_x,0)) : x_id+n_idx_x +1, np.max((y_id-n_idx_y,0)) : y_id+n_idx_y +1]

        node_ids_in_square = node_ids_in_square.flatten()

        return node_ids_in_square