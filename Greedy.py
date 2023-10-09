import numpy as np

def maxCutSolverGreedy(numVertices, edgeWeigths):
        
    "Start with creating initial cut"
    n = numVertices
    variables = np.random.choice([-1, 1], size=n) 
    
    "create copy of input"
    copyw = edgeWeigths.copy()
    variables_1 = variables.copy()
    variables_2 = variables.copy()
    
    "initialize intarations and solution"
    iterations = 0
    best_solution = 0
    while True:
        "find highest weigth in edgeWeigths and location ,GREEDY (and remove both instances of weight)"
        loca = np.unravel_index(np.argmax(copyw), copyw.shape)
        copyw[loca] = 0
        copyw[loca[1],loca[0]]
        if variables[loca[0]]==variables[loca[1]]:  
            "if highest weight is not in cut, calculate result for relocating either of the nodes"
            variables_1 [loca[0]] = variables_1 [loca[0]] * -1
            variables_2 [loca[1]] = variables_2 [loca[1]] * -1
            now = np.sum(0.25*edgeWeigths[i,j]*(1-variables[i]*variables[j]) for i in range(n) for j in range(n) )
            new_1 = np.sum(0.25*edgeWeigths[i,j]*(1-variables_1[i]*variables_1[j]) for i in range(n) for j in range(n))
            new_2 = np.sum(0.25*edgeWeigths[i,j]*(1-variables_2[i]*variables_2[j]) for i in range(n) for j in range(n))
            solutions = np.array([now, new_1, new_2])
            index = np.argmax(solutions)
            if np.max(solutions) > best_solution:
                best_solution = np.max(solutions)
                "record best solution "
                
            " change cut acordingly"
            if index == 1:
                variables = variables_1.copy()
            if index == 2:
                variables = variables_2.copy()
            if np.sum(copyw)==0:
                "finished when the copy of edgeWeigths is empty"
                break
        
        
        iterations += 1
    return variables