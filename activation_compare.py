import csv
from typing import List
from locale import setlocale, LC_ALL, str as lstr

from neural_network import Network, Entry, sigmoid, tanh, relu


# use Polish (system) locale (comma as decimal point, for proper Excel formatting)
setlocale(LC_ALL, '')


# prepare the network
net = Network([4, 4, 3], 0.02)
# store the generated net data for comparison
net_data = net.export_data()


with open("bezdekIris.data", 'r') as file:
    samples: List[Entry] = []
    for row in csv.reader(file):
        iris_type = row[4]
        iris_outputs: List[int] = [0, 0, 0]
        if iris_type == "Iris-setosa":
            iris_outputs[0] = 1
        elif iris_type == "Iris-versicolor":
            iris_outputs[1] = 1
        elif iris_type == "Iris-virginica":
            iris_outputs[2] = 1
        samples.append(Entry([float(n) for n in row[0:4]], iris_outputs))


def train(activation_f):
    net.import_data(net_data)  # restore original for comparison purposes
    net.set_activation_f(activation_f)
    errors_data = []
    for i, error in enumerate(net.teach_loop(samples), start=1):
        if i > 100:
            break
        errors_data.append(lstr(error))  # use locale
    return errors_data


funcs = (sigmoid, tanh, relu)
data_funs = []

for f in funcs:
    print(f.__name__)
    data_funs.append(train(f))

# transpose for excel
data_funs = list(zip(*data_funs))

with open("neural_data.csv", 'w') as file:
    writer = csv.writer(file, delimiter=';')  # delimit with semicolons
    writer.writerows(data_funs)
