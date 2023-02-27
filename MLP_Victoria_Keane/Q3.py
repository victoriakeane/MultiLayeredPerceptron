from mlp import MLP 
import numpy as np 
import random
import matplotlib.pyplot as plt

num_inputs = 4
num_outputs = 1
num_hidden = 20

max_epochs = 10000
learning_rate = [0.0001, 0.005, 0.01, 0.02, 0.3]

NN = MLP(num_inputs, num_hidden, num_outputs)
NN.get_random_weights()

inputs = []
array = []
count = 0
outputs = []

for i in range(500):
    temp = []
    result = 0
    for j in range(0, 4):
        temp.append(round(random.uniform(-1.0, 1.0), 3))
        result += temp[j]
    array.append(temp)
    # compute sin
    outputs.append(np.sin([temp[0] - temp[1] + temp[2] - temp[3]]))
inputs = np.array(array)



with open('q3_h20_tanh.txt', 'w') as f:
    f.write("Inputs: " + str(num_inputs) + "\n Hidden: " + str(num_hidden) + "\n Outputs : " + str(num_outputs) + "\n Max Epochs: " + str(max_epochs) +  "\n Learning Rate: " + str(learning_rate))
    for i in range(len(learning_rate)):
        error_array = []
        x = learning_rate[i]
        f.write("\n Learning Rate: " + str(x))
        f.write("\n train set epoch \n")
        for j in range(0, max_epochs):
            forward = NN.forward(inputs[:400], "tanh")
            error = NN.backward(inputs[:400], outputs[:400], "tanh", x)
            update = NN.update_weights()

            if j == 100 or j == 500 or j == 1000  or j % (max_epochs/5) == 0 or j == 10000:
                f.write(' \n Error at Epoch: ' + str(j) + ' is ' + str(error))
        f.write("\n")
        f.write("\n Test Results \n")
        results_array = []
        target_array = []
        for k in range(400, 500):
            NN.forward(inputs[k], 'tanh')
            f.write("\n target: " + str(outputs[k]))
            f.write("\t result: " + str(NN.O))
            results_array.append(NN.O)
            target_array.append(outputs[k])
    plt.show()
    plt.plot(learning_rate, error_array, label='Error')
    plt.xlabel('Learning Rate')
    plt.ylabel('Error')
    plt.title('Error to learning rate Hidden=10 || sigmoid')
    plt.legend()
    plt.show()
    
