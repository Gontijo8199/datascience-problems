import numpy as np
import matplotlib.pyplot as plt

'''
Problem: Fit a Line to Data Using Linear Regression

    Generate a synthetic dataset of x and y values:
        x is a set of 10 random numbers between 0 and 10.
        y=2x+3 with some added random noise.
        
    Use the Normal Equation to calculate the best-fit line
    
'''

def generator():
    np.random.seed(42) 
    noise = np.random.randn(10) # can be negative
    x = np.random.rand(10) * 10
    y = 2 * x + 3 + noise
    
    X = np.vstack((x, np.ones_like(x))).T 
    theta = np.linalg.inv(X.T @ X) @ X.T @ y  # (X^T X)^-1 X^T y 
    
    slope, intercept = theta

    print(f"Slope: {slope}, Intercept: {intercept}")

    plt.scatter(x, y, label="Data", color="blue")
    plt.plot(x, slope * x + intercept, label="Best-fit Line", color="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Linear Regression with NumPy")
    plt.show()

generator()

