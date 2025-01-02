'''
Problem: Implement K-Nearest Neighbors (KNN) for Classification

    Create a small synthetic dataset with two classes:
        Features: 2D points (x, y).
        Labels: Class 0 or Class 1.

    Write a function to predict the class of a new point based on its kk nearest neighbors from the dataset:
        Use the Euclidean distance to find the neighbors.
        Predict the class by majority vote.

    Test the implementation by predicting the class of a few new points and visualize the decision boundaries. 

    Note: I used linear regression to visualize the flow of the new points compared to both class 0 and class 1 cluster.
'''
 
import numpy as np
import matplotlib.pyplot as plt



np.random.seed(42)

class_0 = np.random.randn(10, 2) + np.array([2, 2])  

class_1 = np.random.randn(10, 2) + np.array([7, 7])  

features = np.vstack((class_0, class_1))  
labels = np.array([0] * 10 + [1] * 10)   

print("Features:\n", features)
print("Labels:\n", labels)

test = np.random.randn(10,2) + np.array([5,5]) 



def knnpredict(newpoint, labels, aux):
   
    distances = np.sqrt(np.sum((aux - newpoint)**2, axis=1))
    
    nearest_indices = np.argsort(distances)[:3]
    
    nearest_labels = labels[nearest_indices]
    
    unique, counts = np.unique(nearest_labels, return_counts=True)
    majority_label = unique[np.argmax(counts)]
    
    return majority_label

def testing(a, b , c):
 for i in range(len(a)):
    prediction = knnpredict(a[i], b, c)
    print(f"Test Point {i}: {a[i]} - Predicted: {prediction} \n", end="") # problema acaba aqui, vou implementar mais a regressao linear destes pontos
        
testing(test, labels, features)

final_total = np.vstack((features, test))

def linear_regression(points):
    x = points[:, 0]
    y = points[:, 1]
    
    n = len(x)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    slope = numerator / denominator
    
    intercept = y_mean - slope * x_mean
    
    return slope, intercept

Z = linear_regression(final_total)

slope = Z[0]
y_intercept = Z[1]

x = np.linspace(0, 10, 400)

y = slope * x + y_intercept

plt.plot(x, y, label=f'y = {slope}x + {y_intercept}', color='yellow')
plt.scatter(class_0[:, 0], class_0[:, 1], color='blue', label='Class 0') 
plt.scatter(class_1[:, 0], class_1[:, 1], color='red', label='Class 1')   
plt.scatter(test[:,0], test[:,1], color = 'green', label = 'New Introduced Points')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Synthetic Dataset for KNN with a linear regression')
plt.legend()
plt.grid(True)
plt.show()

