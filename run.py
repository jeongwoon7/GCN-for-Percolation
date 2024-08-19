import pandas as pd
import torch
from modules import train_and_evaluate
from modules import bs, pcs
import os
from multiprocessing import Pool

torch.set_num_threads(1)
num_processors = 21
lr = 1E-6
weight_decay = 0.0
epochs = 100
loss_diff_criterion = 1E-6
max_epochs_without_improvement = 1000000
step_size = 10000000
gamma = 0.2
num_run = 1

path="./saved_models"
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)


def process_batch(pcs_batch):
    args = [(num_run, pc, bs, lr, epochs, loss_diff_criterion, max_epochs_without_improvement, weight_decay, step_size, gamma) for pc in pcs_batch]
    with Pool(processes=min(num_processors, len(pcs_batch))) as pool:
        results = pool.starmap(train_and_evaluate, args)
    return results


def main():

    batch_size = 21
    results = []

    for i in range(0, len(pcs), batch_size):
        pcs_batch = pcs[i:i+batch_size]
        results.extend(process_batch(pcs_batch))

    # Temporary DataFrames to store the results of this run
    df_a = pd.DataFrame({pc: [a] for pc, a, _, _ in results})
    df_a.loc[len(df_a)] = {pc: std for pc, _, std, _ in results}
    df_performance = pd.DataFrame({pc: pd.Series(performance_ratio) for pc, _, _, performance_ratio in results})

    df_a.to_excel('./df_p_graph.xlsx', index=False)
    df_performance.to_excel('./df_performance.xlsx', index=False)


if __name__ == '__main__':
    main()
