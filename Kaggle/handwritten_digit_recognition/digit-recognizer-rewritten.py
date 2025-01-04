import numpy as np
import pandas as pd
import pickle
import signal
import sys

'''
 
    https://www.kaggle.com/code/rgontijof/digit-recognizer-1
 
    https://www.kaggle.com/competitions/digit-recognizer
    
    made with love, by Rafael Gontijo

 
'''

# Funções auxiliares
def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def oneHot(x):
    n_classes = x.max() + 1
    one_hot_x = np.zeros((x.size, n_classes))
    one_hot_x[np.arange(x.size), x] = 1
    return one_hot_x

def predict(x):
    return np.argmax(x, axis=1)

def accuracy(predic, y):
    return np.mean(predic == y)

# Classe da Rede Neural
class NeuralNetwork:
    def __init__(self, input_neurons, hidden1_neurons, hidden2_neurons, output_neurons):
        # Inicialização de pesos e biases
        self.w1 = np.random.randn(input_neurons, hidden1_neurons)  * np.sqrt(2.0 / hidden1_neurons)
        self.b1 = np.zeros((1, hidden1_neurons))
        self.w2 = np.random.randn(hidden1_neurons, hidden2_neurons) * np.sqrt(2.0 / hidden2_neurons)
        self.b2 = np.zeros((1, hidden2_neurons))
        self.w3 = np.random.randn(hidden2_neurons, output_neurons) * np.sqrt(2.0 / output_neurons)
        self.b3 = np.zeros((1, output_neurons))
    
    def FowardPropagation(self, X):
        self.input2hidden1 = np.dot(X, self.w1) + self.b1
        self.output_hidden1 = ReLU(self.input2hidden1)
        
        self.hidden1_2_hidden2 = np.dot(self.output_hidden1, self.w2) + self.b2
        self.output_hidden2 = ReLU(self.hidden1_2_hidden2)
        
        self.hidden2_output = np.dot(self.output_hidden2, self.w3) + self.b3
        self.output_output = softmax(self.hidden2_output)
        
        return self.output_output

    def BackPropagation(self, X, y, learningrate):
        size_y = y.shape[0]
        y = oneHot(y)
        
        error_output = self.output_output - y
        w3_error = np.dot(self.output_hidden2.T, error_output) / size_y
        b3_error = np.sum(error_output, axis=0, keepdims=True) / size_y

        hidden2_error = np.dot(error_output, self.w3.T) * ReLU_derivative(self.hidden1_2_hidden2)
        w2_error = np.dot(self.output_hidden1.T, hidden2_error) / size_y
        b2_error = np.sum(hidden2_error, axis=0, keepdims=True) / size_y

        hidden1_error = np.dot(hidden2_error, self.w2.T) * ReLU_derivative(self.input2hidden1)
        w1_error = np.dot(X.T, hidden1_error) / size_y
        b1_error = np.sum(hidden1_error, axis=0, keepdims=True) / size_y

        # Atualização dos pesos e biases
        self.w3 -= learningrate * w3_error
        self.b3 -= learningrate * b3_error
        self.w2 -= learningrate * w2_error
        self.b2 -= learningrate * b2_error
        self.w1 -= learningrate * w1_error
        self.b1 -= learningrate * b1_error

    def save_model(self, filename='model_weights.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump({
                'w1': self.w1,
                'b1': self.b1,
                'w2': self.w2,
                'b2': self.b2,
                'w3': self.w3,
                'b3': self.b3
            }, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='model_weights.pkl'):
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
                self.w1 = model_data['w1']
                self.b1 = model_data['b1']
                self.w2 = model_data['w2']
                self.b2 = model_data['b2']
                self.w3 = model_data['w3']
                self.b3 = model_data['b3']
                print(f"Model loaded from {filename}")
        except FileNotFoundError:
            print(f"No saved model found at {filename}. Starting with random weights.")
    
    def Training(self, X, y, learningrate, iterations):
        self.load_model()
        for i in range(iterations):
            self.FowardPropagation(X)
            self.BackPropagation(X, y, learningrate)
            
            if i % 10 == 0:
                p = predict(self.output_output)
                acc = accuracy(p, y)
                print(f"Iteration {i}: Accuracy = {acc * 100:.2f}%")
                
        self.save_model()
        
    def predict_and_save(self, input_file, output_file):
        test_data = pd.read_csv(input_file).to_numpy() / 255.0
        output = self.FowardPropagation(test_data)
        predictions = predict(output)

        submission = pd.DataFrame({"ImageId": np.arange(1, len(predictions) + 1), "Label": predictions})
        submission.to_csv(output_file, index=False)




# Função principal
def main():
    
    nn = NeuralNetwork(input_neurons=784, hidden1_neurons=124, hidden2_neurons=64, output_neurons=10)
    
    '''
    data = pd.read_csv('train.csv').to_numpy()
    np.random.shuffle(data)
    
    X = data[:, 1:] / 255.0  # Normalização dos dados de entrada
    y = data[:, 0]  # Rótulos

    nn.Training(X, y, learningrate=0.2, iterations=500)
    
    output = nn.FowardPropagation(self, X)
    predict = nn.predict(output)
    accuracy = nn.accuracy(predict, y)
   
    print(f"Accuracy = {accuracy * 100:.2f}%")

    
    ''' 
    
    nn.predict_and_save('test.csv', 'submission.csv')
    
    def handle_interrupt(signal, frame):
        print("\nKeyboard interrupt received. Saving model...")
        nn.save_model()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, handle_interrupt)


if __name__ == '__main__':
    main()
