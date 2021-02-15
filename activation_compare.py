import csv
from typing import List
from locale import setlocale, LC_ALL, str as lstr

from neural_network import Network, Entry, sigmoid, tanh, relu


# use Polish (system) locale (comma as decimal point, for proper Excel formatting)
setlocale(LC_ALL, '')

funcs = (sigmoid, tanh, relu)
# read data
with open("bezdekIris.data", 'r') as file:
    samples = []
    for row in csv.reader(file):
        iris_type = row[4]
        iris_outputs = [0, 0, 0]
        if iris_type == "Iris-setosa":
            iris_outputs[0] = 1
        elif iris_type == "Iris-versicolor":
            iris_outputs[1] = 1
        elif iris_type == "Iris-virginica":
            iris_outputs[2] = 1
        samples.append(Entry([float(n) for n in row[0:4]], iris_outputs))


def train_network():
    # prepare the network
    net = Network([4, 4, 3], 0.02)
    # store the generated net data for comparison
    net_data = net.export_data()

    def train(activation_f):
        net.import_data(net_data)  # restore original for comparison purposes
        net.set_activation_f(activation_f)
        errors_data = []
        for i, error in enumerate(net.teach_loop(samples), start=1):
            errors_data.append(error)
            if i >= cycles:
                break
        return errors_data

    funs_data = []
    for f in funcs:
        print(f.__name__)
        funs_data.append(train(f))
    return funs_data


times = 10
cycles = 100
# repeat for average
avg_data = train_network()
for _ in range(times-1):
    data = train_network()
    # consolidate
    for i, new_func_data, avg_func_data in zip(range(len(funcs)), data, avg_data):
        avg_data[i] = [sum(d) for d in zip(new_func_data, avg_func_data)]
# divide and convert to polish notation (use locale)
for i, func_data in enumerate(avg_data):
    avg_data[i] = [lstr(error / times) for error in func_data]
# transpose for excel
avg_data = list(zip(*avg_data))
# save
with open("neural_data.csv", 'w') as file:
    writer = csv.writer(file, delimiter=';')  # delimit with semicolons
    writer.writerows(avg_data)
