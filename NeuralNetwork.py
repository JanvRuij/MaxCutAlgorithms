from tensorflow.keras.models import load_model
import numpy as np

def maxCutSolverNeuralNetwork(numVertices, edgeWeigths, model):
         
    "Start with creating initial cut"
    n = numVertices
    variables = np.random.choice([-1, 1], size=n) 
    variables_copy = variables.copy()

    "create copy of input"
    copyw = edgeWeigths.copy()
    
    "initialize intarations and solution"
    iterations = 0
    best_solution = 0
    while True:
        # Find the highest three numbers in de matrix. We need index 5 becouse of the symmetry of the matrix.
        indices = np.argpartition(copyw, -5, axis=None)[-5:]
        # Convert the flat indices to 2D indices
        loca = np.unravel_index(indices, copyw.shape)

        #Create a 3x3 matrix filled with zeros
        matrix = np.zeros((3, 3))

        # Fill the upper triangular part
        matrix[0, 1] = copyw[loca[0][0]][loca[1][0]]
        matrix[0, 2] = copyw[loca[0][2]][loca[1][2]]
        matrix[1, 2] = copyw[loca[0][4]][loca[1][4]]

        # Mirror the upper triangular part to the lower triangular part
        matrix = matrix + matrix.T - np.diag(matrix.diagonal())

        # Set the copy values to 0 and their symmetric versions to 0
        copyw[loca[0][0]][loca[1][0]] = copyw[loca[1][0]][loca[0][0]] = 0
        copyw[loca[0][2]][loca[1][2]] = copyw[loca[1][2]][loca[0][2]] = 0
        copyw[loca[0][4]][loca[1][4]] = copyw[loca[1][4]][loca[0][4]] = 0

        # Make predictions using the loaded model
        predictions = model.predict(matrix.reshape(1,9),verbose = 0)
        
        result = np.zeros(3)
        for i in range(len(predictions[0])):
            if i > 0:
                result[i] = 1
            else:
                result[i] = -1

        "if highest weight is not in cut, calculate result for relocating either of the nodes"
        variables_copy[loca[0][0]] = result[0]
        variables_copy[loca[0][2]] = result[1]
        variables_copy[loca[0][4]] = result[2]

        new = np.sum(0.25*edgeWeigths*(1 - np.outer(variables_copy,variables_copy)))
        if new > best_solution:
            best_solution = new
            variables = variables_copy.copy()
            "record best solution "
        else:
            "If it is not an improvement, we do not want a changed copy"
            variables_copy = variables.copy()
        if np.sum(copyw)==0:
            "finished when the copy of edgeWeigths is empty"
            break
    
        iterations += 1          
    
    return variables

# wij =  np.random.rand(10, 10)
# wij[np.tril_indices(10)] = 0   # set diagonal and below to zero"
# wij = wij + wij.T 

# # Specify the path to the saved model file
# model_path = 'Trained-NeuralNetwork'
# from tensorflow.keras.models import load_model
# # Load the model
# loaded_model = load_model(model_path)
# print(maxCutSolverNeuralNetwork(10,wij,loaded_model))