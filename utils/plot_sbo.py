import numpy as np
import matplotlib.pyplot as plt
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
import seaborn as sns
from matplotlib.patches import FancyArrowPatch


def norm2phy(x_norm, phy_bounds, chord_bounds):
    
    x_norm = np.atleast_2d(x_norm)

    x_phy = phy_bounds[:,0] + (phy_bounds[:,1] - phy_bounds[:,0]) * x_norm          # relative chord distance

    # Calculate Endpoint
    ce_DS = x_phy[:,0] + x_phy[:,1] * (chord_bounds[1] - x_phy[:,0])                                
    ce_SS = x_phy[:,3] + x_phy[:,4] * (chord_bounds[1] - x_phy[:,3])  

    x_phy[:,1] = ce_DS
    x_phy[:,4] = ce_SS

    return x_phy


def get_opt_iteration_plot(sbo_obj, funcSpecs, phy_bounds, chord_bounds, T_mode, start_iter = 0, fig_ax = None, n_type = None, y_anchor = 1.0):

    probObj = sbo_obj.problemObj
    data = probObj.GetStoredValues(funcSpecs)

    x_norm = data["varValues"]["x"]
    x_phy = norm2phy(x_norm, phy_bounds, chord_bounds)
    y = -data["val"]

    best_idx = []
    for elem in probObj.historyBest:
        best_idx.append(elem["combIdxs"])
    best_idx = np.array(best_idx)

    all_idx = np.arange(x_phy.shape[0])
    flag = np.zeros(x_phy.shape[0], dtype = bool)
    flag[best_idx] = 1 
    no_optim_idx = all_idx[~flag]
    optim_idx = all_idx[flag]

    all_idx = np.arange(x_phy.shape[0])
    iter_idx = all_idx[all_idx > start_iter]
    optim_idx = optim_idx[optim_idx > start_iter]
    no_optim_idx = no_optim_idx[no_optim_idx > start_iter]

    # ObjVal Plot
    if fig_ax is None:
        fig, ax = plt.subplots(5,1)
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]
    
    ax[0].scatter(no_optim_idx, y[no_optim_idx], marker = "o", c = "grey", alpha = 0.2, s = 5)
    h_opt = ax[0].scatter(optim_idx, y[optim_idx], marker = "*", c = color_cycle[2], alpha = 0.5)
    ax[0].set_ylabel("Obj Fun")
    ax[0].tick_params(labelbottom = False)
    sns.despine(ax=ax[0], offset=5)

    h_c_ds = ax[1].scatter(optim_idx, x_phy[optim_idx,1], marker = "^", c = color_cycle[0], label = "ce_ds", alpha = 0.5)
    ax[1].scatter(optim_idx, x_phy[optim_idx,0], marker = "v", c = color_cycle[0], label = "cs_ds", alpha = 0.5)
    ax[1].scatter(no_optim_idx, x_phy[no_optim_idx,1], marker = "^", c = "grey", alpha = 0.2, s = 5)
    ax[1].scatter(no_optim_idx, x_phy[no_optim_idx,0], marker = "v", c = "grey", alpha = 0.2, s = 5)
    ax[1].tick_params(labelbottom = False)
    # ax[1].legend(loc = "lower left")
    ax[1].set_ylabel("Chord")
    sns.despine(ax=ax[1], offset=5)

    h_c_ss = ax[2].scatter(optim_idx, x_phy[optim_idx,4], marker = "^", c = color_cycle[1], label = "ce_ss", alpha = 0.5)
    ax[2].scatter(optim_idx, x_phy[optim_idx,3], marker = "v", c = color_cycle[1], label = "cs_ss", alpha = 0.5)
    ax[2].scatter(no_optim_idx, x_phy[no_optim_idx,4], marker = "^", c = "grey", alpha = 0.2, s = 5)
    ax[2].scatter(no_optim_idx, x_phy[no_optim_idx,3], marker = "v", c = "grey", alpha = 0.2, s = 5)
    ax[2].tick_params(labelbottom = False)
    # ax[2].legend(loc = "lower left")
    ax[2].set_ylabel("Chord")
    sns.despine(ax=ax[2], offset=5)

    h_d_ds = ax[3].scatter(optim_idx, x_phy[optim_idx,2], marker = "o", c = color_cycle[0], label = "d_ds", alpha = 0.5)
    ax[3].scatter(no_optim_idx, x_phy[no_optim_idx,2], marker = "o", c = color_cycle[0], alpha = 0.2, s = 5)
    h_d_ss = ax[3].scatter(optim_idx, x_phy[optim_idx,5], marker = "o", c = color_cycle[1], label = "d_ss", alpha = 0.5)
    ax[3].scatter(no_optim_idx, x_phy[no_optim_idx,5], marker = "o", c = color_cycle[1], alpha = 0.2, s = 5)
    ax[3].tick_params(labelbottom = False)
    # ax[3].legend(loc = "lower left")
    ax[3].set_ylabel("Angle")
    sns.despine(ax=ax[3], offset=5)

    if T_mode[0]:
        h_T_ds = ax[4].scatter(optim_idx, x_phy[optim_idx,6], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.5)
        ax[4].scatter(no_optim_idx, x_phy[no_optim_idx,6], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.2, s = 5)
        h_T_ss = ax[4].scatter(optim_idx, x_phy[optim_idx,7], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.5)
        ax[4].scatter(no_optim_idx, x_phy[no_optim_idx,7], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.2, s = 5)
        ax[4].set_xlabel("Model evaluations")
        ax[4].set_ylabel("Actuator mode")
    else:
        h_T_ds = ax[4].scatter(optim_idx, np.ones(len(optim_idx)) * T_mode[1], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.5)
        ax[4].scatter(no_optim_idx, np.ones(len(no_optim_idx)) * T_mode[1], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.2, s = 5)
        h_T_ss = ax[4].scatter(optim_idx, np.ones(len(optim_idx)) * T_mode[2], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.5)
        ax[4].scatter(no_optim_idx, np.ones(len(no_optim_idx)) * T_mode[2], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.2, s = 5)
        # ax[4].legend(loc = "lower left")
        ax[4].set_xlabel("Model evaluations")
        ax[4].set_ylabel("Actuator mode")

    if n_type is not None:
        n_doe = n_type[0]
        n_EI = n_type[1]
        n_TR = n_type[2]
        add_right_arrow(ax[4], start_iter, n_doe, -800, 'DOE')
        add_double_arrow(ax[4], n_doe, n_doe + n_EI, -800, 'EI')
        add_double_arrow(ax[4], n_doe + n_EI, n_doe + n_EI + n_TR, -800, 'TR')
    sns.despine(ax=ax[4], offset=5)

    empty_handle = plt.Line2D([0], [0], marker='o', color='w', label='', markerfacecolor='w', markersize=10)

    # custom_line2 = Line2D([], [], color='black', label = "10 - 90 \% Quantile")
    # custom_box_patch = Patch(facecolor='lightblue', edgecolor='Black', label='25 - 75 \% Quantile')
    # custom_line = Line2D([0], [0], color='red', lw=1, linestyle='-', label = "Median")
    fig.legend(handles = [h_opt, empty_handle, h_c_ds, h_c_ss, h_d_ds, h_d_ss, h_T_ds, h_T_ss], labels = ["Opt. Delta phi","", "chord ps", "chord ss", "Orientation ps", "Orientation ss", "act. mode ps", "act. mode ss"], loc='upper center', bbox_to_anchor=(0.5, y_anchor), ncol=4, frameon = False)


    return ax


def plot_balde_sim(x, objFunVal, T, xlabel, output_folder, file_name):
    n = len(objFunVal)
    iter = np.arange(n) + 1

    fig, ax = plt.subplots(5,1, figsize = [6,6])
    h_opt = ax[0].scatter(iter, objFunVal, marker = "*", c = color_cycle[2], alpha = 0.5)
    ax[0].set_ylabel("delta_phi")
    ax[0].tick_params(labelbottom = False)
    sns.despine(ax=ax[0], offset=5)

    h_c_ds = ax[1].scatter(iter, x[:,1], marker = "^", c = color_cycle[0], label = "ce_ds", alpha = 0.5)
    ax[1].scatter(iter, x[:,0], marker = "v", c = color_cycle[0], label = "cs_ds", alpha = 0.5)
    ax[1].set_ylabel("chord ds")
    ax[1].tick_params(labelbottom = False)
    ax[1].set_yticks([0.1, 0.5, 0.9])
    ax[1].set_ylim([0.0, 1.0])
    ax[1].axhline(y=0.9, color='grey', linestyle='--', linewidth=1)
    ax[1].axhline(y=0.1, color='grey', linestyle='--', linewidth=1)
    sns.despine(ax=ax[1], offset=5)

    h_c_ss = ax[2].scatter(iter, x[:,4], marker = "^", c = color_cycle[1], label = "ce_ss", alpha = 0.5)
    ax[2].scatter(iter, x[:,3], marker = "v", c = color_cycle[1], label = "ce_ss", alpha = 0.5)
    ax[2].set_ylabel("chord ss")
    ax[2].tick_params(labelbottom = False)
    ax[2].set_yticks([0.1, 0.5, 0.9])
    ax[2].set_ylim([0.0, 1.0])
    ax[2].axhline(y=0.9, color='grey', linestyle='--', linewidth=1)
    ax[2].axhline(y=0.1, color='grey', linestyle='--', linewidth=1)
    sns.despine(ax=ax[2], offset=5)

    h_d_ds = ax[3].scatter(iter, x[:,2], marker = "o", c = color_cycle[0], label = "d_ds", alpha = 0.5)
    h_d_ss = ax[3].scatter(iter, x[:,5], marker = "o", c = color_cycle[1], label = "d_ss", alpha = 0.5)
    ax[3].set_ylabel("angle")
    ax[3].tick_params(labelbottom = False)
    ax[3].set_yticks([0.0, 90.0, 180])
    sns.despine(ax=ax[3], offset=5)

    h_T_ds = ax[4].scatter(iter, T[:,0], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.5)
    h_T_ss = ax[4].scatter(iter, T[:,1], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.5)
    ax[4].set_ylabel("Actuator mode")
    ax[4].set_xticks(iter)
    ax[4].set_xticklabels(xlabel)
    sns.despine(ax=ax[4], offset=5)
    # 

    y_anchor = 1.05
    empty_handle = plt.Line2D([0], [0], marker='o', color='w', label='', markerfacecolor='w', markersize=10)
    fig.legend(handles = [h_opt, empty_handle, h_c_ds, h_c_ss, h_d_ds, h_d_ss, h_T_ds, h_T_ss], labels = ["Opt. Delta phi","", "chord ps", "chord ss", "Orient. ps", "Orient. ss", "act. mode ps", "act. mode ss"], loc='upper center', bbox_to_anchor=(0.5, y_anchor), ncol=4, frameon = False)

    # Save plot
    fig.tight_layout()
    fig.savefig(f'{output_folder}/{file_name}.pdf', format = 'pdf', bbox_inches = 'tight', pad_inches = 0.0)

def get_opt_iteration_plot_GA(target, c_bounds, ax = None):

    data_GE = get_GE_log(target, c_bounds)
    if target == "DeltaPhi":
        y = data_GE[:,9]
        x_phy = data_GE[:,0:8]
    elif target == "Kappa":
        y = data_GE[:,8]
        x_phy = data_GE[:,0:8]

    idx_GE = np.arange(len(y))

    if ax is None:
        fig, ax = plt.subplots(4,1)

    ax[0].scatter(idx_GE, y, marker = "o", c = '#2ca02c', label = target)
    ax[0].set_ylabel("Obj Fun")
    ax[0].tick_params(labelbottom = False)

    ax[1].scatter(idx_GE, x_phy[:,0], marker = "v", c = '#1f77b4', label = "cs_ds")
    ax[1].scatter(idx_GE, x_phy[:,1], marker = "^", c = '#1f77b4', label = "ce_ds")
    ax[1].scatter(idx_GE, x_phy[:,3], marker = "v", c = '#ff7f0e', label = "cs_ss")
    ax[1].scatter(idx_GE, x_phy[:,4], marker = "^", c = '#ff7f0e', label = "ce_ss")
    ax[1].tick_params(labelbottom = False)
    ax[1].legend(loc = "lower left")
    ax[1].set_ylabel = "Chord"

    ax[2].scatter(idx_GE, x_phy[:,2], marker = "o", c = '#1f77b4', label = "d_ds")
    ax[2].scatter(idx_GE, x_phy[:,5], marker = "o", c = '#ff7f0e', label = "d_ss")
    ax[2].tick_params(labelbottom = False)
    ax[2].legend(loc = "lower left")
    ax[2].set_ylabel = "Angle"

    ax[3].scatter(idx_GE, x_phy[:,6], marker = "o", c = '#1f77b4', label = "T_ds")
    ax[3].scatter(idx_GE, x_phy[:,7], marker = "o", c = '#ff7f0e', label = "T_ss")
    ax[3].legend(loc = "lower left")
    ax[3].set_ylabel = "Actuator mode"
    ax[3].set_xlabel = "Iteration"

    return ax  

def compare_sbo_GE(sbo_obj, funcSpecs, target, c_bounds, ax = None):

    probObj = sbo_obj.problemObj
    data = probObj.GetStoredValues(funcSpecs)
    y_sbo = -data["val"]

    data_GE = get_GE_log(target, c_bounds)
    if target == "DeltaPhi":
        y_GE = data_GE[:,9]
    elif target == "Kappa":
        y_GE = data_GE[:,8]

    idx_GE = np.arange(len(y_GE))
    idx_sbo = np.arange(len(y_sbo))

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(idx_GE, y_GE, marker = "o", c = '#1f77b4', label = "Obj Fun: GA", alpha = 0.5)
    ax.scatter(idx_sbo, y_sbo, marker = "o", c = '#ff7f0e', label = "Obj Fun: SBO", alpha = 0.5)

    return ax

def add_double_arrow(ax, start_x, end_x, y_position, label):
    # Create a double arrow to represent the interval
    arrow_length = 0.2  # Adjust this for arrow length
    arrow = FancyArrowPatch((start_x, y_position), (end_x, y_position),
                             arrowstyle='<|-|>', color='black', lw=1,
                             mutation_scale=10, shrinkA= 0.0, shrinkB= 0.0)
    ax.add_patch(arrow)
    
    # Add the label in the middle of the arrow
    ax.text((start_x + end_x) / 2, y_position + 50, label, ha='center', va='bottom')

def add_right_arrow(ax, start_x, end_x, y_position, label):
    # Create a double arrow to represent the interval
    arrow_length = 0.2  # Adjust this for arrow length
    arrow = FancyArrowPatch((start_x, y_position), (end_x, y_position),
                             arrowstyle='-|>', color='black', lw=1,
                             mutation_scale=10, shrinkA= 0.0, shrinkB= 0.0)
    ax.add_patch(arrow)
    
    # Add the label in the middle of the arrow
    ax.text((start_x + end_x) / 2, y_position + 50, label, ha='center', va='bottom')

def plot_opt_iter(candidate_x, candidate_q, phy_bounds, chord_bounds, n_x, model_calls, T_mode, det_opt, n_doe = None, n_EI = None, n_EM = None, n_TR = None, fig_ax = None):
    # Plot the evolution 
    # candidate_x = np.array(candidate_x)
    candidate_x_phy = norm2phy(candidate_x, phy_bounds= phy_bounds[:n_x,:], chord_bounds= chord_bounds)
    model_calls_EI = np.array(model_calls)
    # candidate_q = np.array(candidate_q)

    n_samples = len(candidate_q)

    best_idx = []

    quant_min = 10000
    for i in range(n_samples):
        if candidate_q[i] < quant_min:
            quant_min = candidate_q[i]
            best_idx.append(i)
    best_idx = np.array(best_idx)

    all_idx = np.arange(n_samples)
    flag = np.zeros(n_samples, dtype = bool)
    flag[best_idx] = 1 
    no_optim_idx = all_idx[~flag]
    optim_idx = all_idx[flag]
    no_optim_model_eval = model_calls[~flag]
    optim_model_eval = model_calls[flag]

    if fig_ax is None: 
        fig, ax = plt.subplots(4,1)
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]
    ax[0].scatter(no_optim_model_eval, -candidate_q[no_optim_idx], marker = "o", c = color_cycle[2], alpha = 0.2, s = 5)
    h_opt = ax[0].scatter(optim_model_eval, -candidate_q[optim_idx], marker = "*", c = color_cycle[2], alpha = 0.5)
    ax[0].tick_params(labelbottom = False)
    sns.despine(ax=ax[0], offset=5)
    ax[0].set_ylabel("$q_{0.05}$")
    ax[0].set_yticks([0.6, 0.8, 1.0])
    ax[0].set_ylim([0.55, 1.1])
    h_detOpt = ax[0].axhline(det_opt, linestyle =  "--", color = "grey", linewidth = 1.0)

    h_c_ds = ax[1].scatter(optim_model_eval, candidate_x_phy[optim_idx,1], marker = "^", c = color_cycle[0], label = "ce_ds", alpha = 0.5)
    ax[1].scatter(optim_model_eval, candidate_x_phy[optim_idx,0], marker = "v", c = color_cycle[0], label = "cs_ds", alpha = 0.5)
    ax[1].scatter(no_optim_model_eval, candidate_x_phy[no_optim_idx,1], marker = "^", c = "grey", alpha = 0.2, s = 5)
    ax[1].scatter(no_optim_model_eval, candidate_x_phy[no_optim_idx,0], marker = "v", c = "grey", alpha = 0.2, s = 5)
    ax[1].tick_params(labelbottom = False)
    ax[1].set_yticks([0.15, 0.45, 0.8])
    ax[1].set_ylim([0.1, 0.85])
    # ax[1].legend(loc = "lower left")
    ax[1].set_ylabel("Chord ($\%$)")
    sns.despine(ax=ax[1], offset=5)

    h_c_ss = ax[2].scatter(optim_model_eval, candidate_x_phy[optim_idx,4], marker = "^", c = color_cycle[1], label = "ce_ss", alpha = 0.5)
    ax[2].scatter(optim_model_eval, candidate_x_phy[optim_idx,3], marker = "v", c = color_cycle[1], label = "cs_ss", alpha = 0.5)
    ax[2].scatter(no_optim_model_eval, candidate_x_phy[no_optim_idx,4], marker = "^", c = "grey", alpha = 0.2, s = 5)
    ax[2].scatter(no_optim_model_eval, candidate_x_phy[no_optim_idx,3], marker = "v", c = "grey", alpha = 0.2, s = 5)
    ax[2].tick_params(labelbottom = False)
    ax[2].set_yticks([0.15, 0.45, 0.8])
    # ax[2].legend(loc = "lower left")
    ax[2].set_ylabel("Chord ($\%$)")
    sns.despine(ax=ax[2], offset=5)

    h_d_ds = ax[3].scatter(optim_model_eval, candidate_x_phy[optim_idx,2], marker = "o", c = color_cycle[0], label = "d_ds", alpha = 0.5)
    ax[3].scatter(no_optim_model_eval, candidate_x_phy[no_optim_idx,2], marker = "o", c = color_cycle[0], alpha = 0.2, s = 5)
    h_d_ss = ax[3].scatter(optim_model_eval, candidate_x_phy[optim_idx,5], marker = "o", c = color_cycle[1], label = "d_ss", alpha = 0.5)
    ax[3].scatter(no_optim_model_eval, candidate_x_phy[no_optim_idx,5], marker = "o", c = color_cycle[1], alpha = 0.2, s = 5)
    # ax[3].legend(loc = "lower left")
    ax[3].set_ylabel("Angle (°)")
    # ax[3].set_xlabel("Model calls")
    ax[3].tick_params(labelbottom = False)
    ax[3].set_yticks([0.0, 90, 180])
    sns.despine(ax=ax[3], offset=5)

    if T_mode[0]:
        h_T_ds = ax[4].scatter(optim_model_eval, candidate_x_phy[optim_idx,6], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.5)
        ax[4].scatter(no_optim_model_eval, candidate_x_phy[no_optim_idx,6], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.2, s = 5)
        h_T_ss = ax[4].scatter(optim_model_eval, candidate_x_phy[optim_idx,7], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.5)
        ax[4].scatter(no_optim_model_eval, candidate_x_phy[no_optim_idx,7], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.2, s = 5)
        ax[4].set_xlabel("Model evaluations")
        ax[4].set_ylabel("Mode (°C)")
        ax[4].set_yticks([0, 500, 1000])
        ax[4].set_ylim([0, 1050])
        sns.despine(ax=ax[4], offset=5)
    else:
        h_T_ds = ax[4].scatter(optim_idx, np.ones(len(optim_idx)) * T_mode[1], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.5)
        ax[4].scatter(no_optim_idx, np.ones(len(no_optim_idx)) * T_mode[1], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.2, s = 5)
        h_T_ss = ax[4].scatter(optim_idx, np.ones(len(optim_idx)) * T_mode[2], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.5)
        ax[4].scatter(no_optim_idx, np.ones(len(no_optim_idx)) * T_mode[2], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.2, s = 5)
        # ax[4].legend(loc = "lower left")
        ax[4].set_xlabel("Model evaluations")
        ax[4].set_ylabel("Mode (°C)")
        ax[4].set_yticks([0, 500, 1000])
        ax[4].set_ylim([0, 1050])
        sns.despine(ax=ax[4], offset=5)

    # if n_doe is not None:
    #     add_right_arrow(ax[4], model_calls[0], n_doe, -800, 'DOE')

    for i in range(5):
        ax[i].grid(True, alpha = 0.3)

    if (n_doe is not None) & (n_EI is not None):
        add_double_arrow(ax[4], n_doe, n_doe + n_EI, 50, 'EI')

    if (n_doe is not None) & (n_EI is not None) & (n_EM is not None):
        add_double_arrow(ax[4], n_doe + n_EI, n_doe + n_EI + n_EM, 50, 'EM')

    if (n_doe is not None) & (n_EI is not None) & (n_EM is not None) & (n_TR is not None):
        add_double_arrow(ax[4], n_doe + n_EI + n_EM, n_doe + n_EI + n_EM + n_TR, 50, 'TR')

    y_anchor = 1.07
    empty_handle = plt.Line2D([0], [0], marker='o', color='w', label='', markerfacecolor='w', markersize=10)
    fig.legend(handles = [h_opt, h_detOpt, h_c_ds, h_c_ss, h_d_ds, h_d_ss, h_T_ds, h_T_ss], labels = ["Opt. $q_{0.05}$","Det. opt", "Act. size PS", "Act. size SS", "Act. orient. PS", "Act. orient. SS", "Act. mode PS", "Act. mode SS"], loc='upper center', bbox_to_anchor=(0.5, y_anchor), ncol=4, frameon = False)
    
    # Print the optimum 
    print("Optimal design point:")
    print(f"cs_ds = {candidate_x_phy[optim_idx[-1], 0]:.5f}        cs_ds_norm = {candidate_x[optim_idx[-1], 0]:.5f}")
    print(f"ce_ds = {candidate_x_phy[optim_idx[-1], 1]:.5f}        ce_ds_norm = {candidate_x[optim_idx[-1], 1]:.5f}")
    print(f"d_ds  = {candidate_x_phy[optim_idx[-1], 2]:.5f}        d_ds_norm = {candidate_x[optim_idx[-1], 2]:.5f}")
    print(f"cs_ss = {candidate_x_phy[optim_idx[-1], 3]:.5f}        cs_ss_norm = {candidate_x[optim_idx[-1], 3]:.5f}")
    print(f"ce_ss = {candidate_x_phy[optim_idx[-1], 4]:.5f}        ce_ss_norm = {candidate_x[optim_idx[-1], 4]:.5f}")
    print(f"d_ss  = {candidate_x_phy[optim_idx[-1], 5]:.5f}        d_ss_norm = {candidate_x[optim_idx[-1], 5]:.5f}\n")
    print(f"T_ds  = {candidate_x_phy[optim_idx[-1], 6]:.5f}        T_ds_norm = {candidate_x[optim_idx[-1], 6]:.5f}\n")
    print(f"T_ss  = {candidate_x_phy[optim_idx[-1], 7]:.5f}        T_ss_norm = {candidate_x[optim_idx[-1], 7]:.5f}\n")

    print("Optimal obj fun value:")
    print(f"Delta Phi = {-candidate_q[optim_idx[-1]]:.3f} \n")

    return fig, ax



def get_opt_iteration_plot_RO(x_norm, obj_val, det_opt, phy_bounds, chord_bounds, T_mode, start_iter = 0, n_doe = 1, fig_ax = None, n_infill = None, y_anchor = 0.95):

    x_phy = norm2phy(x_norm, phy_bounds, chord_bounds)
    y = -obj_val

    idx = np.arange(n_doe, len(y))

    y_max = np.max(y[:n_doe])
    best_idx = []
    for i in idx:
        if y[i] > y_max:
            y_max = y[i]
            best_idx.append(i)
    best_idx = np.array(best_idx)

    all_idx = np.arange(x_phy.shape[0])
    flag = np.zeros(x_phy.shape[0], dtype = bool)
    flag[best_idx] = 1 
    no_optim_idx = all_idx[~flag]
    optim_idx = all_idx[flag]

    all_idx = np.arange(x_phy.shape[0])
    iter_idx = all_idx[all_idx > start_iter]
    optim_idx = optim_idx[optim_idx > start_iter]
    no_optim_idx = no_optim_idx[no_optim_idx > start_iter]

    # ObjVal Plot
    if fig_ax is None:
        fig, ax = plt.subplots(5,1)
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]
    ax[0].scatter(no_optim_idx, y[no_optim_idx], marker = "o", c = "grey", alpha = 0.2, s = 5)
    h_opt = ax[0].scatter(optim_idx, y[optim_idx], marker = "*", c = color_cycle[2], alpha = 0.5)
    ax[0].set_ylabel("Obj Fun")
    ax[0].tick_params(labelbottom = False)
    ax[0].axhline(det_opt)
    sns.despine(ax=ax[0], offset=5)

    h_c_ds = ax[1].scatter(optim_idx, x_phy[optim_idx,1], marker = "^", c = color_cycle[0], label = "ce_ds", alpha = 0.5)
    ax[1].scatter(optim_idx, x_phy[optim_idx,0], marker = "v", c = color_cycle[0], label = "cs_ds", alpha = 0.5)
    ax[1].scatter(no_optim_idx, x_phy[no_optim_idx,1], marker = "^", c = "grey", alpha = 0.2, s = 5)
    ax[1].scatter(no_optim_idx, x_phy[no_optim_idx,0], marker = "v", c = "grey", alpha = 0.2, s = 5)
    ax[1].tick_params(labelbottom = False)
    # ax[1].legend(loc = "lower left")
    ax[1].set_ylabel("Chord")
    sns.despine(ax=ax[1], offset=5)

    h_c_ss = ax[2].scatter(optim_idx, x_phy[optim_idx,4], marker = "^", c = color_cycle[1], label = "ce_ss", alpha = 0.5)
    ax[2].scatter(optim_idx, x_phy[optim_idx,3], marker = "v", c = color_cycle[1], label = "cs_ss", alpha = 0.5)
    ax[2].scatter(no_optim_idx, x_phy[no_optim_idx,4], marker = "^", c = "grey", alpha = 0.2, s = 5)
    ax[2].scatter(no_optim_idx, x_phy[no_optim_idx,3], marker = "v", c = "grey", alpha = 0.2, s = 5)
    ax[2].tick_params(labelbottom = False)
    # ax[2].legend(loc = "lower left")
    ax[2].set_ylabel("Chord")
    sns.despine(ax=ax[2], offset=5)

    h_d_ds = ax[3].scatter(optim_idx, x_phy[optim_idx,2], marker = "o", c = color_cycle[0], label = "d_ds", alpha = 0.5)
    ax[3].scatter(no_optim_idx, x_phy[no_optim_idx,2], marker = "o", c = color_cycle[0], alpha = 0.2, s = 5)
    h_d_ss = ax[3].scatter(optim_idx, x_phy[optim_idx,5], marker = "o", c = color_cycle[1], label = "d_ss", alpha = 0.5)
    ax[3].scatter(no_optim_idx, x_phy[no_optim_idx,5], marker = "o", c = color_cycle[1], alpha = 0.2, s = 5)
    ax[3].tick_params(labelbottom = False)
    # ax[3].legend(loc = "lower left")
    ax[3].set_ylabel("Angle")
    sns.despine(ax=ax[3], offset=5)

    if T_mode[0]:
        h_T_ds = ax[4].scatter(optim_idx, x_phy[optim_idx,6], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.5)
        ax[4].scatter(no_optim_idx, x_phy[no_optim_idx,6], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.2, s = 5)
        h_T_ss = ax[4].scatter(optim_idx, x_phy[optim_idx,7], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.5)
        ax[4].scatter(no_optim_idx, x_phy[no_optim_idx,7], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.2, s = 5)
        ax[4].set_xlabel("Iteration")
        ax[4].set_ylabel("Actuator mode")
    else:
        h_T_ds = ax[4].scatter(optim_idx, np.ones(len(optim_idx)) * T_mode[1], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.5)
        ax[4].scatter(no_optim_idx, np.ones(len(no_optim_idx)) * T_mode[1], marker = "s", c = color_cycle[0], label = "T_ds", alpha = 0.2, s = 5)
        h_T_ss = ax[4].scatter(optim_idx, np.ones(len(optim_idx)) * T_mode[2], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.5)
        ax[4].scatter(no_optim_idx, np.ones(len(no_optim_idx)) * T_mode[2], marker = "s", c = color_cycle[1], label = "T_ss", alpha = 0.2, s = 5)
        # ax[4].legend(loc = "lower left")
        ax[4].set_xlabel("Iteration")
        ax[4].set_ylabel("Actuator mode")

    if n_infill is not None:
        n_EI = n_infill[0]
        n_EM = n_infill[1]
        n_TR = n_infill[2]
        add_right_arrow(ax[4], start_iter, n_doe, -800, 'DOE')
        add_double_arrow(ax[4], n_doe, n_doe + n_EI, -800, 'EI')
        add_double_arrow(ax[4], n_doe + n_EI, n_doe + n_EI + n_EM, -800, 'EM')
        add_double_arrow(ax[4], n_doe + n_EI + n_EM, n_doe + n_EI + n_EM + n_TR, -800, 'TR')
    sns.despine(ax=ax[4], offset=5)

    empty_handle = plt.Line2D([0], [0], marker='o', color='w', label='', markerfacecolor='w', markersize=10)

    # custom_line2 = Line2D([], [], color='black', label = "10 - 90 \% Quantile")
    # custom_box_patch = Patch(facecolor='lightblue', edgecolor='Black', label='25 - 75 \% Quantile')
    # custom_line = Line2D([0], [0], color='red', lw=1, linestyle='-', label = "Median")
    fig.legend(handles = [h_opt, empty_handle, h_c_ds, h_c_ss, h_d_ds, h_d_ss, h_T_ds, h_T_ss], labels = ["Opt. Delta phi","", "chord ps", "chord ss", "Orient. ps", "Orient. ss", "act. mode ps", "act. mode ss"], loc='upper center', bbox_to_anchor=(0.5, y_anchor), ncol=4, frameon = False)

    x_opt = x_phy[optim_idx[-1]]
    y_opt = y[optim_idx]

    print("Optimal design point:")
    print(f"cs_ds = {x_phy[optim_idx[-1], 0]:.5f}        cs_ds_norm = {x_norm[optim_idx[-1], 0]:.5f}")
    print(f"ce_ds = {x_phy[optim_idx[-1], 1]:.5f}        ce_ds_norm = {x_norm[optim_idx[-1], 1]:.5f}")
    print(f"d_ds  = {x_phy[optim_idx[-1], 2]:.5f}        d_ds_norm = {x_norm[optim_idx[-1], 2]:.5f}")
    print(f"cs_ss = {x_phy[optim_idx[-1], 3]:.5f}        cs_ss_norm = {x_norm[optim_idx[-1], 3]:.5f}")
    print(f"ce_ss = {x_phy[optim_idx[-1], 4]:.5f}        ce_ss_norm = {x_norm[optim_idx[-1], 4]:.5f}")
    print(f"d_ss  = {x_phy[optim_idx[-1], 5]:.5f}        d_ss_norm = {x_norm[optim_idx[-1], 5]:.5f}\n")

    print("Optimal obj fun value:")
    print(f"Delta Phi = {y[optim_idx[-1]]:.3f} \n")

    return ax

def get_opt_iteration_objFun(obj_val, start_iter = 0, n_doe = 1, fig_ax = None, n_infill = None, deter_opt = None):

    y = -obj_val

    idx = np.arange(n_doe, len(y))

    y_max = np.max(y[:n_doe])
    best_idx = []
    for i in idx:
        if y[i] > y_max:
            y_max = y[i]
            best_idx.append(i)
    best_idx = np.array(best_idx)

    all_idx = np.arange(len(y))
    flag = np.zeros(len(y), dtype = bool)
    flag[best_idx] = 1 
    no_optim_idx = all_idx[~flag]
    optim_idx = all_idx[flag]

    all_idx = np.arange(len(y))
    iter_idx = all_idx[all_idx > start_iter]
    optim_idx = optim_idx[optim_idx > start_iter]
    no_optim_idx = no_optim_idx[no_optim_idx > start_iter]

    # ObjVal Plot
    if fig_ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = fig_ax[0]
        ax = fig_ax[1]
    h1 = ax.scatter(no_optim_idx, y[no_optim_idx], marker = "o", c = "grey", alpha = 0.2, s = 5, label = "Obj. Fun eval")
    h2 = ax.scatter(optim_idx, y[optim_idx], marker = "*", c = color_cycle[2], alpha = 0.5, label = "Obj. Fun optimum")
    ax.set_ylabel("Obj Fun")
    ax.set_xlabel("Model evaluations")

    if deter_opt is not None:
        h3 = ax.hlines(deter_opt, xmin = start_iter, xmax = len(y), colors = "dimgray", linestyle = "--", label = "Determ. opt.")



    if n_infill is not None:
        n_EI = n_infill[0]
        n_TR = n_infill[1]
        add_right_arrow(ax, start_iter, n_doe, -0.2, 'DOE')
        add_double_arrow(ax, n_doe, n_doe + n_EI, -0.2, 'EI')
        add_double_arrow(ax, n_doe + n_EI, n_doe + n_EI + n_TR, -0.2, 'TR')
    sns.despine(ax=ax, offset=0)

    # ax.legend(frameon = False)

    # custom_line2 = Line2D([], [], color='black', label = "10 - 90 \% Quantile")
    # custom_box_patch = Patch(facecolor='lightblue', edgecolor='Black', label='25 - 75 \% Quantile')
    # custom_line = Line2D([0], [0], color='red', lw=1, linestyle='-', label = "Median")
    fig.legend(handles = [h1, h2, h3], labels = ["Obj. Fun eval","Obj. Fun optimum", "Determ. opt."], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon = False)


    return ax
