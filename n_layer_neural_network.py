__author__ = 'tan_nguyen'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

def generate_data():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y

def generate_circles_data():
    '''
    generate circle data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(200, noise=0.01)
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

########################################################################################################################
########################################################################################################################
# YOUR ASSSIGMENT STARTS HERE
# FOLLOW THE INSTRUCTION BELOW TO BUILD AND TRAIN A N-LAYER NEURAL NETWORK
########################################################################################################################
########################################################################################################################

class Layer(object):
    """
    This class builds for a single layer in the neural network
    """
    def __init__(self, nn_input_dim, nn_output_dim, last_layer = 0, actFun_type='tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_output_dim: output dimension
        :param actFun_type: type of activation function. 3 options: 'tanh', 'sigmoid', 'relu'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        :param last_layer: input last layer of the network
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_output_dim = nn_output_dim
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda
        self.last_layer = last_layer

        # initialize the weights and biases in the network
        np.random.seed(seed)
        self.W = np.random.randn(self.nn_input_dim, self.nn_output_dim) / np.sqrt(self.nn_input_dim)
        self.b = np.zeros((1, self.nn_output_dim))

    def feedforward(self, X, actFun):
        '''
        feedforward builds the feedforward steps for a single layer in the network
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        self.z = np.dot(X, self.W) + self.b

        # Intermediate Layer:
        if self.last_layer == 0:
            self.a = actFun(self.z)
        # Last Layer:
        else:
            exp_scores = np.exp(self.z)
            self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        return None

########################################################################################################################
########################################################################################################################

class DeepNeuralNetwork(object):
    """
    This class builds and trains a neural network
    """
    def __init__(self, nn_input_dim, nn_hidden_dim, nn_output_dim, num_layers, actFun_type='Tanh', reg_lambda=0.01, seed=0):
        '''
        :param nn_input_dim: input dimension
        :param nn_hidden_dim: the number of hidden units
        :param nn_output_dim: output dimension
        :param num_layers: input layer
        :param actFun_type: type of activation function. 3 options: 'Tanh', 'Sigmoid', 'ReLU'
        :param reg_lambda: regularization coefficient
        :param seed: random seed
        '''
        self.nn_input_dim = nn_input_dim
        self.nn_hidden_dim = nn_hidden_dim
        self.nn_output_dim = nn_output_dim
        self.num_layers = num_layers
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        # Create "n" Layers:
        self.LayerList = []
        for count in range(num_layers - 1):
            if count == 0:
                x = Layer(nn_input_dim=self.nn_input_dim, nn_output_dim=self.nn_hidden_dim, last_layer=0,
                          actFun_type=self.actFun_type, seed=count)
            elif (count < num_layers - 2):
                x = Layer(nn_input_dim=self.nn_hidden_dim, nn_output_dim=self.nn_hidden_dim, last_layer=0,
                          actFun_type=self.actFun_type, seed=count)
            else:
                x = Layer(nn_input_dim=self.nn_hidden_dim, nn_output_dim=self.nn_output_dim, last_layer=1,
                          actFun_type=self.actFun_type, seed=count)
            self.LayerList.append(x)

    def actFun(self, z, type):
        '''
        actFun computes the activation functions
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: activations
        '''

        # YOU IMPLMENT YOUR actFun HERE
        if type == 'Tanh':
            activations = np.tanh(z)
        elif type == 'Sigmoid':
            activations = 1 / (1 + np.exp(-z))
        elif type == 'ReLU':
            activations = z * (z > 0)
        else:
            print('Invalid activation function type.')

        return activations

    def diff_actFun(self, z, type):
        '''
        diff_actFun compute the derivatives of the activation functions wrt the net input
        :param z: net input
        :param type: Tanh, Sigmoid, or ReLU
        :return: the derivatives of the activation functions wrt the net input
        '''

        # YOU IMPLEMENT YOUR diff_actFun HERE
        if  type == 'Tanh':
            activations = np.tanh(z)
            diff_activations =  1 - (activations * activations)
        elif type == 'Sigmoid':
            activations = 1 / (1 + np.exp(-z))
            diff_activations = activations * (1 - activations)
        elif type == 'ReLU':
            diff_activations = np.array(z)
            diff_activations[diff_activations < 0] = 0
            diff_activations[diff_activations > 0] = 1
        else:
            print('Invalid activation function type.')

        return diff_activations

    def feedforward(self, X, actFun):
        '''
        feedforward builds a n-layer neural network and computes the two probabilities,
        one for class 0 and one for class 1
        :param X: input data
        :param actFun: activation function
        :return:
        '''

        # YOU IMPLEMENT YOUR feedforward HERE
        for count in range(self.num_layers - 1):
            if count == 0:
                self.LayerList[count].feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            elif (count < self.num_layers - 2):
                self.LayerList[count].feedforward(self.LayerList[count - 1].a,
                                                  lambda x: self.actFun(x, type=self.actFun_type))
            else:
                self.LayerList[count].feedforward(self.LayerList[count - 1].a,
                                                  lambda x: self.actFun(x, type=self.actFun_type))
                self.probs = self.LayerList[count].probs

        return None

    def calculate_loss(self, X, y):
        '''
        calculate_loss compute the loss for prediction
        :param X: input data
        :param y: given labels
        :return: the loss for prediction
        '''
        num_examples = len(X)
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        # Calculating the loss

        # YOU IMPLEMENT YOUR CALCULATION OF THE LOSS HERE
        data_loss = np.sum(-np.log(self.probs[range(num_examples), y]))
        # data_loss =

        # Add regulatization term to loss
        tempsum = 0
        for count in range(self.num_layers - 1):
            tempsum += np.sum(np.square(self.LayerList[count].W))
        data_loss += self.reg_lambda / 2 * tempsum
        return (1. / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
        return np.argmax(self.probs, axis=1)

    def backprop(self, X, y):
        '''
        backprop run backpropagation to compute the gradients used to update the parameters in the backward step
        :param X: input data
        :param y: given labels
        :return: dL/dW, dL/db
        '''

        # IMPLEMENT YOUR BACKPROP HERE
        dW = []
        db = []
        delta = []
        for count in range(self.num_layers - 1):
            dW.append([])
            db.append([])
            delta.append([])

        num_examples = len(X)
        delta[self.num_layers - 2] = self.probs
        delta[self.num_layers - 2][range(num_examples), y] -= 1

        for count in range(self.num_layers - 2, -1, -1):
            if count == self.num_layers - 2:
                dW[count] = (self.LayerList[count - 1].a.T).dot(delta[count])
                db[count] = np.sum(delta[count], axis=0, keepdims=True)

            elif count == 0:
                delta[count] = delta[count + 1].dot(self.LayerList[count + 1].W.T) * self.diff_actFun(
                    self.LayerList[count].z, self.actFun_type)
                dW[count] = (X.T).dot(delta[count])
                db[count] = np.sum(delta[count], axis=0, keepdims=True)
            else:
                delta[count] = delta[count + 1].dot(self.LayerList[count + 1].W.T) * self.diff_actFun(
                    self.LayerList[count].z, self.actFun_type)
                dW[count] = (self.LayerList[count - 1].a.T).dot(delta[count])
                db[count] = np.sum(delta[count], axis=0, keepdims=True)
        
        return dW, db

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        '''
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        '''
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X, lambda x: self.actFun(x, type=self.actFun_type))
            # Backpropagation
            dW, db = self.backprop(X, y)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            for count in range(self.num_layers - 1):
                dW[count] += self.reg_lambda * self.LayerList[count].W
                self.LayerList[count].W += -epsilon * dW[count]
                self.LayerList[count].b += -epsilon * db[count]

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def visualize_decision_boundary(self, X, y):
        '''
        visualize_decision_boundary plot the decision boundary created by the trained network
        :param X: input data
        :param y: given labels
        :return:
        '''
        plot_decision_boundary(lambda x: self.predict(x), X, y)

def main():
    # # generate and visualize Make-Moons dataset
    # X, y = generate_data()
    # # generate and visualize Make-circles dataset
     X, y = generate_circles_data()
     # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
     # plt.show()

     model = DeepNeuralNetwork(nn_input_dim=2, nn_hidden_dim = 7, nn_output_dim=2, num_layers = 3, actFun_type='ReLU')
     model.fit_model(X,y, epsilon = 0.001)
     model.visualize_decision_boundary(X,y)

if __name__ == "__main__":
    main()
