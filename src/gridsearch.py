import itertools as it
import time
from joblib import Parallel, delayed
from main_explain import main_explain

dataset_list = ["MUTAG"]
lr_list = [0.01, 0.1, 0.5, 1]
epoch_list = [300, 500]
beta_list = [0, 0.1, 0.5]
gamma_list = [0, 0.1, 0.5]
momentum_list = [0, 0.5, 0.9]
edge_del_list = [False, True]
edge_add_list = [False, True]
bernoulli_list = [True, False]
delta_list = [True, False]
cem_list = ["PP"]
cuda_list = [False]
rand_init_list = [True, False]

hyperpar_combo = {"dataset_id": dataset_list,
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
                  "cuda": cuda_list,
                  "rand_init": rand_init_list}

dict_keys, dict_vals = zip(*hyperpar_combo.items())
combo_list = list(it.product(*dict_vals))

task_list = []

for i, combo in enumerate(combo_list):

    combo_dict = {dict_keys[i]: combo[i] for i in range(len(dict_keys))}

    # edge_add in the orig formulation is identical to edge_add + edge_del
    if combo_dict["cem_mode"] is None and combo_dict["edge_add"] and combo_dict["edge_del"]\
            and not combo_dict["delta"]:
        continue

    task_list.append(delayed(main_explain)(**combo_dict))

start_time = time.time()
print("Starting gridsearch: 0/{}".format(len(task_list)))
Parallel(n_jobs=-2, verbose=11)(task_list)

end_time = time.time()
time_mins = (end_time-start_time)//60

print("Grid search performed in: {} minutes".format(time_mins))
