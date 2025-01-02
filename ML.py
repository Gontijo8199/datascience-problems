import numpy as np
'''

Make a simple neural network using only numpy library
    
This includes back and foward propagation 

'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    def forward(self, X):
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = sigmoid(self.output_layer_input)
        
        return self.output_layer_output

    def backward(self, X, y, learning_rate):
        output_error = y - self.output_layer_output
        output_delta = output_error * sigmoid_derivative(self.output_layer_output)
        
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_layer_output)
        
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y, learning_rate)
            if (epoch + 1) % 1000 == 0:
                loss = np.mean(np.square(y - self.output_layer_output))
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss}')

X = np.array([[0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 0, 1, 0],
              [1, 0, 1, 0, 1]])

y = np.array([[0, 0],
              [1, 1],
              [1, 0],
              [0, 1]])

nn = NeuralNetwork(input_size=5, hidden_size=5, output_size=2)
nn.train(X, y, epochs=10000, learning_rate=0.1)

print("Predictions after training:")
print(nn.forward(X))
