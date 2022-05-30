import itertools as it
import time
import torch
from torch.multiprocessing import set_start_method, freeze_support
from main_explain import setup_env, server_explain

dataset_id = "syn1"
lr_list = [0.01, 0.1, 0.5, 1]
epoch_list = [300, 500]
beta_list = [0, 0.1, 0.5]
gamma_list = [0.1, 0.5]
momentum_list = [0, 0.5, 0.9]
edge_del_list = [False, True]
edge_add_list = [False, True]
bernoulli_list = [True, False]
delta_list = [True]
cem_list = ["PP"]
rand_init_list = [0, 0.01, 0.5]

def gridsearch():

    dataset, model, device = setup_env(dataset_id, cuda=True)

    hyperpar_combo = {"dataset": [dataset],
                      "model": [model],
                      "lr": lr_list,
                      "num_epochs": epoch_list,
                      "beta": beta_list,
                      "gamma": gamma_list,
                      "n_momentum": momentum_list,
                      #"edge_del": edge_del_list,
                      #"edge_add": edge_add_list,
                      "bernoulli": bernoulli_list,
                      "delta": delta_list,
                      "cem_mode": cem_list,
                      "device": [device],
                      "rand_init": rand_init_list,
                      "n_workers": [2]}

    dict_keys, dict_vals = zip(*hyperpar_combo.items())
    combo_list = list(it.product(*dict_vals))
    num_combos = len(combo_list)

    print("Starting gridsearch")
    start_time = time.time()

    for i, combo in enumerate(combo_list):

        combo_dict = {dict_keys[i]: combo[i] for i in range(len(dict_keys))}

        # edge_add in the orig formulation is identical to edge_add + edge_del
        if combo_dict["cem_mode"] is None and combo_dict["edge_add"] and combo_dict["edge_del"]\
                and not combo_dict["delta"]:
            continue

        server_explain(**combo_dict)
        print(f"Task {i+1}/{num_combos} completed")

    end_time = time.time()
    time_mins = (end_time-start_time)//60

    print("Grid search performed in: {} minutes".format(time_mins))


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn")
    gridsearch()
