import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB, LinExpr

def maxCutBinarySolver(numVertices, edgeWeigths):
    
    n = numVertices 
    wij = edgeWeigths

    # empty variables
    variables = []
    
    # Create a new model
    m = Model("mip1")
    m.Params.OutputFlag = 0  # Set to 0 to disable solver output
    # Create variables
    for i in range(n):
        variables.append(m.addVar(vtype=GRB.BINARY, name="Player: {}".format(i)))
        variables[i] = variables[i]*2-1                                              #makes bianry -1 or 1
     
        
    """
     Set objective
    
    """
        
    # Create a list to store the objective terms
    objective_terms = []
    
    # Create a linear expression for the objective
    for i in range(n):
        for j in range(n):
            term = 0.25*wij[i,j]*(1-variables[i]*variables[j])
            objective_terms.append(term)
    
    # Sum up the objective terms
    objective_expr = LinExpr()
    for term in objective_terms:
        objective_expr += term
    
    # Set the objective to minimize the expression
    m.setObjective(objective_expr, GRB.MAXIMIZE)
    
    """
     SOLVE
    
    """
    
    # Optimize model
    m.optimize()
    cut= np.empty(len(m.getVars()))
    
    for i, v in enumerate(m.getVars()):
        cut[i] = v.X
    return cut*2-1
