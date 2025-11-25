import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

values = [-5, -2, -1, 0, 1, 2, 5]
results = {x: sigmoid(x) for x in values}

results
