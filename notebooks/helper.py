import os
import pickle 
from matplotlib.patches import FancyArrowPatch
import torch 

# def pad_rows(rows):
#     n = max(r.numel() for r in rows)
#     return torch.stack([
#         torch.cat([r, r[-1].expand(n - r.numel())]) if r.numel() < n else r
#         for r in rows
#     ])

def pad_rows(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Pad each 2D tensor to the maximum number of rows by repeating
    its last row.
    """
    if not tensors:
        return []

    if any(t.ndim != 2 for t in tensors):
        raise ValueError("All tensors must be 2-dimensional.")

    if any(t.shape[0] == 0 for t in tensors):
        raise ValueError("Tensors must contain at least one row.")

    n_columns = tensors[0].shape[1]
    if any(t.shape[1] != n_columns for t in tensors):
        raise ValueError("All tensors must have the same number of columns.")

    max_rows = max(t.shape[0] for t in tensors)

    return [
        torch.cat(
            [
                tensor,
                tensor[-1:].expand(max_rows - tensor.shape[0], -1),
            ],
            dim=0,
        )
        for tensor in tensors
    ]

def get_batch_sim(directory, key):
    matched_files = []
    for filename in os.listdir(directory):
        if filename.startswith(key) and filename.endswith("_data.pkl"):
            matched_files.append(filename)

    print(matched_files)

    best_q = []
    best_x = []

    len_model_calls = 0

    # n_EI_max = 0
    # n_EM_max = 0 
    # n_TR_max = 0

    for file in matched_files:
        loaded_objects = []
        with open(f'{directory}/{file}', 'rb') as f:
            try:
                while True:
                    loaded_objects.append(pickle.load(f))
            except EOFError:
                pass

        ro_data, model, gp_model, args = loaded_objects
        best_x_list =  [ro_data.doe["cand_x"]] + ro_data.EI_infill["best_x"] + ro_data.EM_infill["best_x"] + ro_data.TR_infill["best_x"]
        # print(best_x_list)
        best_x_list = torch.row_stack(best_x_list)
        best_q_list = [ro_data.doe["cand_J"]] + ro_data.EI_infill["best_J"] + ro_data.EM_infill["best_J"] + ro_data.TR_infill["best_J"]

        best_q.append(torch.tensor(best_q_list).reshape(-1,1))

        best_x.append(best_x_list)

        model_calls = ro_data.doe["model_calls"] + ro_data.EI_infill["model_calls"] + ro_data.EM_infill["model_calls"] + ro_data.TR_infill["model_calls"]
        if len(model_calls) > len_model_calls: 
            model_calls_cumsum = torch.cumsum(torch.tensor(model_calls), dim=0)
            len_model_calls = len(model_calls)
            n_EI_max = ro_data.EI_infill["n"]
            n_EM_max = ro_data.EM_infill["n"]
            n_TR_max = ro_data.TR_infill["n"]

    ref_x, ref_q = model.load_opt(measure = args.measure, alpha = args.alpha, tau = args.tau)

    best_x_padded = torch.stack(pad_rows(best_x), dim = 1)
    best_q_padded = torch.column_stack(pad_rows(best_q))
    # return best_q, best_x1, best_x2, model_calls_cumsum, n_EI, n_TR
    return best_q_padded, best_x_padded, model_calls_cumsum, n_EI_max, n_EM_max, n_TR_max, ref_x, ref_q

def add_double_arrow(ax, start_x, end_x, y_position, label):
    # Create a double arrow to represent the interval
    arrow_length = 0.2  # Adjust this for arrow length
    arrow = FancyArrowPatch((start_x, y_position), (end_x, y_position),
                             arrowstyle='<|-|>', color='black', lw=1,
                             mutation_scale=10, shrinkA= 0.0, shrinkB= 0.0)
    ax.add_patch(arrow)
    
    # Add the label in the middle of the arrow
    ax.text((start_x + end_x) / 2, y_position + 0.02, label, ha='center', va='bottom')

    return ax

def add_infill_segements(ax, model_calls_cumsum, n_EI):
    x_pos = (model_calls_cumsum[n_EI]+model_calls_cumsum[n_EI+1])/2
    ax.axvline(x=x_pos, color='grey', linestyle='--')
    y_max = ax.get_ylim()[1]
    y_min = ax.get_ylim()[0]
    print(y_max)
    y_90 = y_min + (y_max - y_min) * 0.9

    x_lim = ax.get_xlim()
    x_diff = x_lim[1] - x_lim[0]

    print(y_90)
    # Add text labels near the vertical line
    ax.text(x_pos - 0.1*x_diff, y_90, "EI", fontsize=10, verticalalignment='center')
    ax.text(x_pos + 0.05*x_diff, y_90, "TR", fontsize=10, verticalalignment='center')

    return ax