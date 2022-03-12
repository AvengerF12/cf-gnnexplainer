import itertools as it
import time
from main_explain import main_explain

dataset_list = ["syn5"]
lr_list = [0.01, 0.5, 1]
epoch_list = [100, 300]
beta_list = [0.1, 0.3, 0.5]
momentum_list = [0, 0.5, 0.7]

hype_par_combo = {"dataset": dataset_list,
                  "lr": lr_list,
                  "num_epochs": epoch_list,
                  "beta": beta_list,
                  "n_momentum": momentum_list}

dict_keys, dict_vals = zip(*hype_par_combo.items())
combo_list = list(it.product(*dict_vals))
num_combos = len(combo_list)

start_time = time.time()

for i, combo in enumerate(combo_list):

    print("{}/{} combos completed".format(i, num_combos))

    combo_dict = {dict_keys[i]: combo[i] for i in range(len(dict_keys))}

    main_explain(**combo_dict)

end_time = time.time()
time_mins = (end_time-start_time)//60

print("Grid search performed in: {} minutes".format(time_mins))
