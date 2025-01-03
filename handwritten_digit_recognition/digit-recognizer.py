import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
 
'''
 
    https://www.kaggle.com/code/rgontijof/digit-recognizer-1
 
    https://www.kaggle.com/competitions/digit-recognizer
 
'''

data = pd.read_csv('train.csv').to_numpy()

np.random.shuffle(data)

ddata = data[:1000].T
dy = ddata[0]

dx = ddata[1:] / 255.0  

datat = data[1000:].T
train_y = datat[0]

train_x = datat[1:] / 255.0  

def nn():
    w1 = np.random.randn(10, 784) * np.sqrt(2.0 / 784)
    b1 = np.zeros((10, 1))
    w2 = np.random.randn(10, 10) * np.sqrt(2.0 / 10)
    b2 = np.zeros((10, 1))

    return w1, w2, b1, b2



def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  
    return e_Z / np.sum(e_Z, axis=0, keepdims=True)


def foward(w1, w2, b1, b2, data):
    c1 = np.dot(w1, data) + b1
    o1 = ReLU(c1)
    c2 = np.dot(w2, o1) + b2
    o2 = softmax(c2)

    return c1,o1,c2,o2

def one_hot(Y):
    n_classes = Y.max() + 1
    one_hot_Y = np.zeros((n_classes, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y


def backwards( c1,o1,c2,o2, w1, w2, y, data):
    a = y.size
    one_hot_y = one_hot(y)

    dz2 = o2 - one_hot_y
    dw2 = np.dot(dz2, o1.T) / a
    db2 = np.sum(dz2, axis=1, keepdims=True) / a

    dz1 = np.dot(w2.T, dz2) * ReLU_derivative(c1)
    dw1 = np.dot(dz1, data.T) / a
    db1 = np.sum(dz1, axis=1, keepdims=True) / a        

    return dw2, dw1, db2, db1

def modify(w1, w2, b1, b2, dw2, dw1, db2, db1, lc):
    w1 -= dw1 * lc
    w2 -= dw2 * lc 
    b1 -= db1 * lc
    b2 -= db2 * lc

    return w1,w2,b1,b2

def predic(x):
    return np.argmax(x, axis=0)
def accur(predic, y):
    return np.mean(predic == y)


def gradient_descent(data, y, lc, aux):
    w1, w2, b1, b2 = nn()
    for i in range(aux):
        c1,o1,c2,o2 = foward(w1, w2, b1, b2, data)
        dw2, dw1, db2, db1 = backwards(c1,o1,c2,o2, w1, w2, y, data)
        w1, w2, b1, b2 = modify(w1, w2, b1, b2, dw2, dw1, db2, db1, lc)

        if i % 10 == 0:
            p = predic(o2)
            ac = accur(p, y)
            print(f"i = {i}: ac = {ac * 100:.2f}%")

    return w1, w2, b1, b2 

w1, w2, b1, b2 = gradient_descent(train_x, train_y, lc=0.2, aux=500)

# post-training

tc1,to1,tc2,to2 =  foward(w1, w2, b1, b2, dx)
pt = predic(to2)
act = accur(pt, dy)
print(f"ac = {act * 100:.2f}%")

# test

data = pd.read_csv('test.csv').to_numpy()

def predict_and_save(w1, w2, b1, b2, input_file, output_file):
    test_data = pd.read_csv(input_file).to_numpy().T / 255.0 
    _, _, _, o2 = foward(w1, w2, b1, b2, test_data)
    predictions = predic(o2)

    submission = pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions})
    submission.to_csv(f"{output_file}", index=False)

predict_and_save(w1, w2, b1, b2, 'test.csv', 'submission.csv')

