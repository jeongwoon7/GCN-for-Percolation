import numpy as np
import random
import pandas as pd
from modules import num_rows, num_cols, node_dim

# Set the number of samples for each p sector eg. (0,0.05), (0.05,0.10), ..., (0.95,1.00)
Ntrain=250
Ntest=150
npcs=21  # pcs = np.linspace(0, 1, npcs)

# Generating random numbers for data generation
def generate_random_numbers(npcs,nsamples):
    pcs = np.linspace(0, 1, npcs)
    step_size = pcs[1]-pcs[0]
    random_numbers = []

    for i in range(npcs-1):
        lower_bound = i * step_size
        upper_bound = (i + 1) * step_size
        # Generate random numbers within the current sector
        sector_numbers = np.random.uniform(low=lower_bound, high=upper_bound, size=nsamples)
        random_numbers.extend(sector_numbers)

    return random_numbers


# Make configurations for training and validation
def make_configurations(npcs,nsamples):

    configurations = []
    occupation_probabilities = generate_random_numbers(npcs,nsamples)

    for p in occupation_probabilities:
        
        lattice = np.zeros((num_rows, num_cols), dtype=int)
        adjacency_matrix = np.zeros((num_rows * num_cols, num_rows * num_cols), dtype=int)
        
        node_features = np.zeros((num_rows * num_cols, node_dim)) 

        # Occupy a site (i,j) with a probability of p
        for i in range(num_rows):
            for j in range(num_cols):
                occupation_status = int(random.random() <= p)
                lattice[i, j] = occupation_status

        # Build the adjacency matrix based on the lattice configuration
        for i in range(num_rows):
            for j in range(num_cols):
                if lattice[i, j] == 1:
                    # Check the status of the neighboring sites
                    if i > 0 and lattice[i - 1, j] == 1:  # Upper neighbor
                        adjacency_matrix[i * num_cols + j, (i - 1) * num_cols + j] = 1
                    if i < num_rows - 1 and lattice[i + 1, j] == 1:  # Lower neighbor
                        adjacency_matrix[i * num_cols + j, (i + 1) * num_cols + j] = 1
                    if j > 0 and lattice[i, j - 1] == 1:  # Left neighbor
                        adjacency_matrix[i * num_cols + j, i * num_cols + (j - 1)] = 1
                    if j < num_cols - 1 and lattice[i, j + 1] == 1:  # Right neighbor
                        adjacency_matrix[i * num_cols + j, i * num_cols + (j + 1)] = 1

        # Build the 20-epochs_from_pc0.55_2nd-dimensional node feature vectors
        for i in range(num_rows):
            for j in range(num_cols):

                occupation_status = lattice[i, j]
                node_features[num_rows*i+j,:] = [occupation_status,0]

                # If boundary
                if i==0:
                    node_features[num_rows*i+j,:] = [occupation_status,1]
                if i==num_rows-1:
                    node_features[num_rows*i+j,:] = [occupation_status,2]

        configuration = {
            'p': p,
            'lattice': lattice.flatten().tolist(),
            'adjacency_matrix': adjacency_matrix.flatten().tolist(),
            'node_features' : node_features.tolist()
        }
        configurations.append(configuration)
            
    return configurations

configurations = make_configurations(npcs,Ntrain)
configurations_test = make_configurations(npcs,Ntest)


# Save the configuration information in CSV files for future usage
df = pd.DataFrame(configurations)
df.to_csv('configurations.csv')

df_test = pd.DataFrame(configurations_test)
df_test.to_csv('configurations_test.csv', index=False)
