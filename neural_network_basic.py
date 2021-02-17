import csv
import random
from time import sleep
from math import exp, inf
from functools import partial
from contextlib import suppress
from collections import namedtuple
from locale import setlocale, LC_ALL, str as lstr
from typing import Union, List, Tuple, Optional, Iterator, Generator

from snake import Game

Entry = namedtuple("Entry", ["inputs", "outputs"])


# use Polish (system) locale (comma as decimal point, for proper Excel formatting)
setlocale(LC_ALL, '')


def sigmoid(value, *, derivative=False):
    if derivative:
        v = sigmoid(value)
        return v * (1 - v)
    else:
        # Overflow prevention
        if value < -709:
            return -1.0
        elif value > 709:
            return 1.0
        return 1 / (1 + exp(-value))


def tanh(value, *, derivative=False):
    if derivative:
        return 1 - pow(tanh(value), 2)
    else:
        # Overflow prevention
        if value < -709:
            return -1.0
        elif value > 709:
            return 1.0
        pos = exp(value)
        neg = exp(-value)
        return (pos - neg) / (pos + neg)


def relu(value, *, derivative=False):
    if derivative:
        if value > 0:
            return 1
        elif value < 0:
            return 0
        else:
            return 0.5
    else:
        return max(0, value)


class IntegrityError(Exception):
    pass


class Network:
    def __init__(self, layer_counts, learning_rate=0.05):
        if len(layer_counts) < 2:
            raise IntegrityError("Can't create network with fewer than 2 Layers!")
        self.layers = []
        layer = None
        for count in layer_counts:
            layer = Layer(self, count, layer)
            self.layers.append(layer)
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.learning_rate = learning_rate
        self.fitness = 0

    def set_activation_f(self, func):
        for layer in self.layers:
            layer.set_activation_f(func)

    def feed_forward(self, inputs):
        self.input_layer.set_inputs(inputs)
        for layer in self.layers:
            layer.feed_forward()
        return self.output_layer.get_outputs()

    def back_propagate(self, targets):
        current_error = self.output_layer.set_errors(targets)
        for layer in reversed(self.layers):
            layer.back_propagate()
        return current_error

    def teach(self, dataset):
        network_error = 0
        for inputs, targets in dataset:
            self.feed_forward(inputs)
            network_error += self.back_propagate(targets)
        return network_error / len(dataset)

    def teach_loop(self, dataset, precision_goal=0.01):
        # validate first
        for learning_set in dataset:
            if not isinstance(learning_set, tuple) or len(learning_set) != 2:
                raise IntegrityError("Incorrect leaning dataset entry!")
            if len(learning_set[0]) != len(self.input_layer):
                raise IntegrityError("Incorrect input dataset entry!")
            if len(learning_set[1]) != len(self.output_layer):
                raise IntegrityError("Incorrect output dataset entry!")
        # teach until the desired error is reached
        network_error = inf
        while network_error > precision_goal:
            network_error = self.teach(dataset)
            yield network_error

    def SP_crossover(self, other):
        cls = type(self)
        n1 = self.export_data()
        n2 = other.export_data()
        assert len(n1) == len(n2)
        n = random.randint(1, len(n1) - 1)
        args = ([len(l) for l in self.layers], self.learning_rate)
        n3, n4 = cls(*args), cls(*args)
        n3.import_data(n1[:n] + n2[n:])
        n4.import_data(n2[:n] + n1[n:])
        return (n3, n4)

    def export_data(self):
        data = []
        for layer in self.layers:
            data.extend(layer.export_data())
        return data

    def import_data(self, data):
        with suppress(TypeError):
            if len(data) != sum(
                (len(n.dendrons) + 1 for l in self.layers for n in l.neurons)
            ):
                raise IntegrityError("Invalid import data size!")
        data_iter = iter(data)
        try:
            for layer in self.layers:
                layer.import_data(data_iter)
        except StopIteration:
            raise IntegrityError("Not enough import data to initialize the network!")


class Layer:
    def __init__(self, network, neuron_count, previous_layer=None):
        self.network = network
        self.neurons = [Neuron(self, previous_layer) for _ in range(neuron_count)]

    def __len__(self):
        return len(self.neurons)

    def set_activation_f(self, func):
        for neuron in self.neurons:
            neuron.set_activation_f(func)

    def set_inputs(self, inputs):
        if len(inputs) != len(self.neurons):
            raise IntegrityError("Incorrect number of inputs!")
        for output, neuron in zip(inputs, self.neurons):
            neuron.set_output(output)

    def set_errors(self, targets):
        if len(targets) != len(self.neurons):
            raise IntegrityError("Incorrect number of outputs!")
        outputs = self.get_outputs()
        errors = [target - output for target, output in zip(targets, outputs)]
        for error, neuron in zip(errors, self.neurons):
            neuron.set_error(error)
        return sum(pow(e, 2) for e in errors)

    def get_outputs(self):
        return [neuron.output for neuron in self.neurons]

    def feed_forward(self):
        for neuron in self.neurons:
            neuron.feed_forward()

    def back_propagate(self):
        for neuron in self.neurons:
            neuron.back_propagate()

    def export_data(self):
        data = []
        for neuron in self.neurons:
            data.extend(neuron.export_data())
        return data

    def import_data(self, data_iter):
        for neuron in self.neurons:
            neuron.import_data(data_iter)


class Neuron:
    def __init__(self, layer, previous_layer=None):
        self.layer = layer
        self.dendrons = []
        self.error = 0
        self.output = 0
        self.bias = 2 * random.random() - 1
        self.activation_f = tanh  # use tanh by default

        if previous_layer is not None:
            for neuron in previous_layer.neurons:
                self.dendrons.append(Connection(neuron))

    def __repr__(self):
        return "Neuron({}, {}, {})".format(self.output, self.bias, self.error)

    def set_activation_f(self, func):
        self.activation_f = func

    def add_error(self, error):
        self.error += error

    def set_error(self, error):
        self.error = error

    def set_output(self, output):
        self.output = output

    def feed_forward(self):
        if self.dendrons:
            s = sum((dendron.get_value() for dendron in self.dendrons), self.bias)
            self.output = self.activation_f(s)

    def back_propagate(self):
        gradient = self.error * self.activation_f(self.output, derivative=True)
        for dendron in self.dendrons:
            dendron.adjust_weight(gradient)
        self.bias += self.layer.network.learning_rate * gradient
        self.error = 0

    def export_data(self):
        # start with the bias
        data = [self.bias]
        # add the weights
        data.extend(d.weight for d in self.dendrons)
        return data

    def import_data(self, data_iter):
        # the first value is the bias
        self.bias = next(data_iter)
        # set weights
        for dendron in self.dendrons:
            dendron.weight = next(data_iter)


class Connection:
    def __init__(self, source_neuron):
        self.source_neuron = source_neuron
        self.weight = 2 * random.random() - 1

    def __repr__(self):
        return "Connection({})".format(self.weight)

    def adjust_weight(self, gradient):
        # pass the gradient to the source neuron
        self.source_neuron.add_error(self.weight * gradient)
        network = self.source_neuron.layer.network
        self.weight += gradient * network.learning_rate * self.source_neuron.output

    def get_value(self):
        return self.source_neuron.output * self.weight


def dist(v1, v2):
    value = v1 - v2
    if value > 100:
        return 10
    if value < 0:
        return 0
    return value / 10


# create the controller
def controller(network, game):
    # extract current inputs
    hx, hy = game.head.position
    ax, ay = game.apple.position
    # inputs = [dist(hx, ax), dist(hy, ay)]
    inputs = [dist(ax, hx), dist(hx, ax), dist(ay, hy), dist(hy, ay)]
    current_dirs = [0, 0, 0, 0]
    current_dirs[game.direction] = 1
    inputs.extend(current_dirs)
    forbidden_dirs = [0, 0, 0, 0]
    # snake body
    for s in game.snake[2:]:
        if hx == s.x:
            if hy + 20 == s.y:
                forbidden_dirs[1] = 1
            elif hy - 20 == s.y:
                forbidden_dirs[3] = 1
        elif hy == s.y:
            if hx + 20 == s.x:
                forbidden_dirs[0] = 1
            elif hx - 20 == s.x:
                forbidden_dirs[2] = 1
    # out of bounds
    wx, wy = game.window.get_size()
    if hx < 20:
        forbidden_dirs[2] = 1
    elif hx > wx - 20:
        forbidden_dirs[0] = 1
    if hy < 20:
        forbidden_dirs[3] = 1
    elif hy > wy - 20:
        forbidden_dirs[1] = 1
    inputs.extend(forbidden_dirs)
    # evaluate
    outputs = network.feed_forward(inputs)
    # set the game outputs
    game.direction = outputs.index(max(outputs))


def evaluate_networks(nets):
    for i, net in enumerate(nets):
        # reset the game to the starting positions
        game.reset()
        # attach the controller
        game.external = partial(controller, net)
        # run the game and calculate the final fitness
        game.run()
        net.fitness = game.score * 1000 - game.steps
        print(f"{i:3}: {net.fitness}")


def select_networks(nets):
    return nets[:len(nets) // 2]


def crossover_networks(nets):
    for n1, n2 in zip(nets[:-1:2], nets[1::2]):  # chunk by two
        nets.extend(n1.SP_crossover(n2))
    return nets


if __name__ == "__main__":
    # init the snake game
    game = Game()
    # run it faaaaaaaast
    game.fps = 600

    # create the population
    population = 100
    net_args = [12, 10, 8, 4]
    networks = [Network(net_args) for _ in range(population)]

    best_net_data = (0, [])
    snake_data = []
    generation = 0
    max_generation = 50
    while True:
        generation += 1
        print(f"Generation: #{generation}")
        # evaluate for fitness
        evaluate_networks(networks)
        # sort with fittest at the top
        networks.sort(key=lambda n: n.fitness, reverse=True)
        best_net = networks[0]
        # save the best net if needed
        print(f"Best fitness: {best_net.fitness}")
        if best_net.fitness > best_net_data[0]:
            best_net_data = (best_net.fitness, best_net.export_data())
            print(f"New best: {best_net.fitness}\n")
        snake_data.append([
            generation,  # current generation
            best_net_data[0],  # current best fitness
            # average fitness of a population
            lstr(sum(n.fitness for n in networks) / len(networks)),
        ])
        if generation >= max_generation:
            break
        # select for the next generation
        networks = select_networks(networks)
        # crossover for the next generation
        networks = crossover_networks(networks)

    # save data for the performance graph
    with open("snake_data.csv", 'w') as file:
        writer = csv.writer(file, delimiter=';')  # delimit with semicolons
        writer.writerows(snake_data)
    # restore the best network
    best_net = Network(net_args)
    best_net.fitness = best_net_data[0]
    best_net.import_data(best_net_data[1])
    # run it
    print(f"\nRunning best network: {best_net.fitness} fitness")
    # attach the controller
    game.external = partial(controller, best_net)
    game.fps = 6
    while not game.window.has_exit:
        # reset the game
        game.reset()
        # run the game
        game.run()
        print(f"Score: {game.score}")
        sleep(1)
