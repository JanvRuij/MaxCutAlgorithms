from Binary import maxCutBinarySolver
import numpy as np
from tqdm import tqdm

#20-20 matrices
n=10

#Dataset met 10.000 matrices en 10.000 oplossingnen
nr_matrices = 10
weights_dataset = []
results_dataset = []

for i in tqdm(range(nr_matrices)):
    wij =  np.random.rand(n, n)
    wij[np.tril_indices(n)] = 0   # set diagonal and below to zero"
    wij = wij + wij.T 
    result = maxCutBinarySolver(n,wij)

    weights_dataset.append(wij.tolist())
    results_dataset.append(result.tolist())


# Open a file in write mode ('w')
with open('10-10x10-Weights_data.txt', 'w') as file:
    # Convert the list to a string and write it to the file
    file.write(str(weights_dataset))
# Open a file in write mode ('w')
with open('10-10x10-Solutions_data.txt', 'w') as file:
    # Convert the list to a string and write it to the file
    file.write(str(results_dataset))