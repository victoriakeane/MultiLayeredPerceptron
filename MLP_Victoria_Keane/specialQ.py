from mlp import MLP 
import numpy as np 
import pandas as pd 


num_inputs = 16
num_outputs = 26
num_hidden = 10
max_epochs = 100
learning_rate = [0.00001]

NN = MLP(num_inputs, num_hidden, num_outputs)
NN.get_random_weights()


letter_df = pd.read_csv("letter-recognition.data", names = ["letterCapital", "xboxHor", "yboxVert",
'width', 'highHeight', 'onpix', 'xbarMean', 'ybarMean', 'x2barMeanVary', 'y2barMeanVary', 'xybarMeanCorr',
'x2ybrMeanMulti', 'Xy2brMeanMulti', 'xegeMean', 'xegvyCorr', 'yegeMean', 'yegvxCorr'])
output_col = letter_df['letterCapital']
a = letter_df.drop(['letterCapital'], axis=1)
inputs = np.array(a) / 20 # keeps inputs as small numbers between 0 : 1, rounded to second decimal

# print(len(letter_df))
# dataset length of 20,000 - I'm going to perform an optimal
# 80/20 train test split (Pareto Principal)
outputs = []
for i in range(10):
    print(output_col[i])
for i in range(20000):
    foo = str(output_col[i])
    outputs.append(ord(foo) - ord('A')) #utf letter value

null_array = np.array([[0 for i in range(26)]for j in range(16000)])
print(null_array)
#  are of zeroes are given some values
for i in range(16000):
    null_array[i][outputs[i]] = 1

outputs = np.array(null_array)

with open('specialQ.txt', 'w') as f:
    f.write("Inputs: " + str(num_inputs) + "\n Hidden: " + str(num_hidden) + "\n Outputs : " + str(num_outputs) + "\n Max Epochs: " + str(max_epochs))
    for i in range(len(learning_rate)):
        x = 0.0
        x = learning_rate[i]
        print("Learning Rate: " + str(x))
        for j in range(0, max_epochs):
            error = 0
            forward = NN.forward(inputs[:16000], "tanh")
            error = NN.backward(inputs[:16000], outputs, "tanh", x)
            update = NN.update_weights()

            if j == 100 or j == 500 or j == 1000  or j % (max_epochs/5) == 0 or j == 10000:
                print(' Error at Epoch: ' + str(j) + ' is ' + str(error))
        other_error = 0
        test_accuracy = 0
        print("Test Results")
        results_array = []
        #  hypothetically  the loops were intended to translate characters to their int values
        for k in range(16001, 20000):
            bar = inputs[k]
            NN.forward(inputs[k], 'tanh')
            vectors = list(NN.O)
            b = vectors.index(max(vectors))
            c = chr(b + ord('A'))
            results_array.append(c)
        # intended to output 26 predicted letters  
        targ_letters = []
        for x in range(16):
            targ_letter = chr(int(bar[x] + ord('A')))
            targ_letters.append(targ_letter)
        result = NN.O 
        result_letters = []  
        for y in range(26):
            result_letter = chr(int(result[y]) + ord('A'))
            result_letters.append(result_letter)

        in_char = chr(int(inputs[k]))
        print("target: " + str(inputs[k]))
        print("result" + str((NN.O)))