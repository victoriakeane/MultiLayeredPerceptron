from mlp import MLP 
import numpy as np 
import matplotlib.pyplot as plt
inputs = np.array([[0, 0], [0, 1], [1,0], [1, 1]])
outputs = np.array([[0], [1], [1], [0]])

num_inputs = 2
num_outputs = 1
num_hidden = 4

max_epochs = 20000

learning_rate = [0.15, 0.25, 0.3, 0.5, 0.6, 0.85]
NN = MLP(num_inputs, num_hidden, num_outputs)
NN.get_random_weights()

with open('xor_results_h4_tanh.txt', 'w') as f:
    f.write("Inputs: " + str(num_inputs) + "\n Hidden: " + str(num_hidden) + "\n Outputs : " + str(num_outputs) + "\n Max Epochs: " + str(max_epochs))
    f.write('\n Targets: ' + str(outputs.T))
    error_array = []
    epoch_array = []
    for i in range(len(learning_rate)):
        x = 0.0
        x = learning_rate[i]
        f.write("\n learning rate: " + str(x))
        for j in range(0, max_epochs+1):
            error = 0
            forward = NN.forward(inputs, "tanh")
            error = NN.backward(inputs, outputs, "tanh", x)
            update = NN.update_weights()

            if j == 100 or j == 500 or j == 1000  or j % (max_epochs/5) == 0 or j == 10000:
                f.write('\n Error at Epoch: ' + str(j) + ' is ' + str(error))
        f.write("\n")
        for i in range(len(inputs)):
            # f.write(str(inputs[i]))
            NN.forward(inputs[i], 'tanh')
            f.write('\n target: ' + str(outputs[i]))
            f.write('\t results: ' + str(NN.O)) 
    plt.plot(learning_rate, error_array, label='Error')
    plt.xlabel('Learning Rate')
    plt.ylabel('Error')
    plt.title('Error to learning rate Hidden=15 || tanh')
    plt.legend()
    plt.show()