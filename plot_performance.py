"""
Visualize the calculated results: performance graph and 2D maps
- Get averaged value over ensemble members (e.g., over 10 runs)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

parent_directory= "/workdir"
count=0
paths=[]
titles=[]

for entry in sorted(os.listdir(parent_directory)):
    if True:
        # Construct the full path of the entry
        entry_path = os.path.join(parent_directory, entry)

        if os.path.isdir(entry_path):
            # Print the name of the subdirectory
            print("Subdirectory:", entry)
            titles.append(entry)
            paths.append(entry_path)
            count += 1

print(entry_path)

all_y_values = []
all_df2_values = []

if count == 1:
    fig, axes = plt.subplots(1, 1, figsize=(3, 10))
    axes = [axes]  # Convert to list to handle the same way as multiple axes
else:
    fig, axes = plt.subplots(1, count, figsize=(30, 10))

for i in range(count):

    df1_path = os.path.join(paths[i], 'df_p_graph.xlsx')
    df2_path = os.path.join(paths[i], 'df_performance.xlsx')

    if not os.path.exists(df1_path) or not os.path.exists(df2_path):
        print(f"Skipping directory {paths[i]} because required .xlsx files are missing.")
        continue

    df1 = pd.read_excel(df1_path)
    df2 = pd.read_excel(df2_path)

    x_values = df1.columns.to_list()
    y_values = df1.iloc[0].values

    df2.columns = x_values

    axes[i].plot(x_values, y_values, 'bo-', label='Mean')
    axes[i].set_xlim(0.0, 1.0)
    axes[i].set_ylim(0.4, 1)
    axes[i].set_title(titles[i])

    # Collecting values
    all_y_values.append(y_values)
    all_df2_values.append(df2)

plt.show()

# Calculating means
mean_y_values = np.mean(all_y_values, axis=0)
std = np.std(all_y_values, axis=0)
mean_df2_values = np.mean(np.array([df.values for df in all_df2_values]), axis=0)

fig, axes = plt.subplots(2, 1, figsize=(5, 10))


axes[0].plot(x_values, mean_y_values, "bo-")
axes[0].fill_between(x_values, mean_y_values - std, mean_y_values + std, alpha=0.2, color='blue', label='Std Dev')
axes[0].set_xlim(0.0, 1)
axes[0].set_ylim(0.0, 1)
axes[0].set_title("epochs=100")

axes[1].imshow(mean_df2_values, cmap='viridis', origin='lower')  # 'cmap' specifies the colormap
axes[1].set_xticks(ticks=np.arange(len(df2.columns)), labels=np.round(df2.columns, 3), fontsize=6)
axes[1].set_yticks(ticks=np.arange(21)-0.5, labels=np.round(np.linspace(0.0,1,21),3), fontsize=6)
axes[1].set_xlabel('parameter pc')
axes[1].set_ylabel('range of p')

plt.tight_layout()
plt.show()
