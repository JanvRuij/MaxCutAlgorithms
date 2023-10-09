import numpy as np
from Greedy import maxCutSolverGreedy
from NeuralNetwork import maxCutSolverNeuralNetwork
from Binary import maxCutBinarySolver
from tensorflow.keras.models import load_model

# Open the file in read mode ('r')
with open('10-20x20-Weights_data.txt', 'r') as file:
    # Read the content of the file
    stored_weights = file.read()


with open('10-20x20-Solutions_data.txt', 'r') as file:
    # Read the content of the file
    stored_results = file.read()


# Convert the stored string back to a list
weights_dataset = eval(stored_weights)
results_dataset = eval(stored_results)

X = np.array(weights_dataset)
Y = np.array(results_dataset)

# Specify the path to the saved model file
model_path = 'Trained-NeuralNetwork'

#Specify size of the weight matrix
n = 20

# Load the model
loaded_model = load_model(model_path)

for i in range(10):
    result_greedy = maxCutSolverGreedy(n,X[i])
    result_NN = maxCutSolverNeuralNetwork(n,X[i],loaded_model)
    result_Binary = maxCutBinarySolver(n,X[i])

    value_greedy = np.sum(0.25 * X[i] * (1 - np.outer(result_greedy, result_greedy)))
    value_NN = np.sum(0.25 * X[i] * (1 - np.outer(result_NN, result_NN)))
    value_Binary = np.sum(0.25 * X[i] * (1 - np.outer(result_Binary, result_Binary)))

    print("Result Greedy: {} --- Result NN: {} --- Result Binary: {}".format(value_greedy,value_NN,value_Binary))

