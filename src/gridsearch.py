import itertools as it
import time
from joblib import Parallel, delayed
from main_explain import main_explain

dataset_list = ["syn5"]
lr_list = [0.01, 0.1, 0.5, 1]
epoch_list = [300, 500]
beta_list = [0.1, 0.5]
momentum_list = [0, 0.5, 0.7, 0.9]

hyperpar_combo = {"dataset": dataset_list,
                  "lr": lr_list,
                  "num_epochs": epoch_list,
                  "beta": beta_list,
                  "n_momentum": momentum_list}

dict_keys, dict_vals = zip(*hyperpar_combo.items())
combo_list = list(it.product(*dict_vals))
num_combos = len(combo_list)

task_list = []

start_time = time.time()

for i, combo in enumerate(combo_list):

    combo_dict = {dict_keys[i]: combo[i] for i in range(len(dict_keys))}
    task_list.append(delayed(main_explain)(**combo_dict))

Parallel(n_jobs=-2)(task_list)

end_time = time.time()
time_mins = (end_time-start_time)//60

print("Grid search performed in: {} minutes".format(time_mins))
