import numpy as np 
import random 

class MLP(): 
    def __init__(self, NI, NH, NO): 
        self.inputs = np.array(NI) #num inputs
        # print(f"inputs {self.inputs}")
        self.hidden = np.array(NH) # num hidden
        self.outputs = NO # num outputs

        self.W1 = list() #lower weights
        self.W2 = list() # higher weights

        self.dW1 = list() # weight change for w1
        self.dW2 = list() # weight change for w2

        self.Z1 = list() #lower activations
        self.Z2 = list() #upper activations

        self.H = [] # hidden neuron values
        self.O = [] # store outputs
        

    def get_random_weights(self):
        # for each input/Output I generate a corresponding weight between 0,1 (small values)
        # each weight sits at just 1 decimal point value 
        # I create a matrix of weights here
        self.W1 = np.array([[round(random.random(), 2)for i in range(self.hidden)]for j in range(self.inputs)])
        print(self.W1)
        self.W2 = np.array([[round(random.random(), 2)for i in range(self.outputs)]for j in range(self.hidden)])
        print(self.W2)
        #settting dW's to 0 
        self.dW1 = np.array([[0 for i in range(self.hidden)]for j in range(self.inputs)])
        self.dW2 = np.array([[0 for i in range(self.outputs)]for j in range(self.hidden)])
        # return self.W1, self.W2, self.dW1, self.dW2


    def forward(self, I, activation):
        self.Z1 = np.dot(I, self.W1)

        if activation == "tanh": 
            self.H = (2 / (1 + np.exp(self.Z1 * -2))) -1
            self.Z2 = np.dot(self.H, self.W2)
            self.O = (2 / (1 + np.exp(self.Z2 * -2))) -1
        else:
            self.H = 1 / (1+np.exp(-self.Z1))
            self.Z2 = np.dot(self.H, self.W2)
            self.O = 1 / (1+np.exp(-self.Z2))

        

    def backward(self, I, targ,learning_method, learning_rate):
        I = np.array(I).T
        if learning_method == "tanh":
            tanh_hidden = (2 / (1 + np.exp(self.Z1 * -2))) - 1
            tanh_output = (2 / (1 + np.exp(self.Z2 * -2))) - 1
            hidden_activation, outer_activation = 1 - (np.power(tanh_hidden, 2)), 1 - (np.power(tanh_output, 2))

        else:
            hidden_activation, outer_activation = np.exp(-self.Z1) / (1 + np.exp(-self.Z1)) ** 2, np.exp(-self.Z2) / (1 + np.exp(-self.Z2)) ** 2

        delta_upper = np.array((targ-self.O) * outer_activation)
        dl = np.dot(delta_upper, self.W2.T)
        delta_lower = np.array(dl * hidden_activation)
        self.dW1 = np.multiply(learning_rate, np.dot(I, delta_lower))
        self.dW2 = np.multiply(learning_rate, np.dot(self.H.T, delta_upper))

        output_error = np.subtract(targ, self.O)
        temp_error = [abs(i) for i in output_error]
        out_err = sum(temp_error)/len(temp_error)
        return out_err

    def update_weights(self):
        self.W1 = np.add(self.W1, self.dW1)
        self.W2 = np.add(self.W2, self.dW2)
        self.dW1 = np.array
        self.dW2 = np.array
         






# np.dot simplifies getting matrix product between hidden and output weights
# weighted sums of all connections to each hidden unit using
# a matrix of weights and multiplying by each input 
# inputs = [[0, 0], [0, 1], [1,0], [1, 1]]
# outputs = [[0], [1], [1], [0]]
# max_epochs = 10000
# learning_rate = 0.7

# nn = MLP(2, 4, 1)
# nn.get_random_weights()
# nn.forward(inputs, "sigmoid")
# nn.backward(inputs, outputs, "sigmoid", learning_rate)

