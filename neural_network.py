import random
from time import sleep
from functools import partial
from contextlib import suppress
from math import exp, inf
from collections import namedtuple
from typing import Union, List, Tuple, Optional, Iterator, Generator

from snake import Game

Entry = namedtuple("Entry", ["inputs", "outputs"])


def tanh(value, *, derivative: bool = False):
    if derivative:
        return 1 - pow(tanh(value), 2)
    else:
        if value < -709:
            return -1.0
        elif value > 709:
            return 1.0
        return (exp(value) - exp(-value)) / (exp(value) + exp(-value))


class IntegrityError(Exception):
    pass


class Network:
    __slots__ = (
        "layers",
        "input_layer",
        "output_layer",
        "learning_rate",
        "momentum_mod",
        "fitness",
        "strikes",
    )

    def __init__(
        self, layer_counts: List[int], learning_rate: float = 0.05, momentum_mod: float = 0.05
    ):
        if len(layer_counts) < 2:
            raise IntegrityError("Can't create network with fewer than 2 Layers!")
        self.layers: List[Layer] = []
        layer = None
        for count in layer_counts:
            layer = Layer(self, count, layer)
            self.layers.append(layer)
        self.input_layer = self.layers[0]
        self.output_layer = self.layers[-1]
        self.learning_rate = learning_rate
        self.momentum_mod = momentum_mod
        self.fitness = 0
        self.strikes = 0

    def set_activation_f(self, func):
        for layer in self.layers:
            layer.set_activation_f(func)

    def feed_forward(self, inputs: List[float]):
        self.input_layer.set_inputs(inputs)
        for layer in self.layers:
            layer.feed_forward()
        return self.output_layer.get_outputs()

    def back_propagate(self, targets: List[float]):
        current_error = self.output_layer.set_errors(targets)
        for layer in reversed(self.layers):
            layer.back_propagate()
        return current_error

    def teach(self, dataset: List[Entry]):
        network_error = 0
        for inputs, targets in dataset:
            self.feed_forward(inputs)
            network_error += self.back_propagate(targets)
        return network_error

    def teach_loop(self, dataset: List[Entry], precision_goal: float = 0.01):
        # validate first
        for learning_set in dataset:
            if not isinstance(learning_set, tuple) or len(learning_set) != 2:
                raise IntegrityError("Incorrect leaning dataset entry!")
            if (
                not isinstance(learning_set[0], list)
                or len(learning_set[0]) != len(self.input_layer)
            ):
                raise IntegrityError("Incorrect input dataset entry!")
            if (
                not isinstance(learning_set[1], list)
                or len(learning_set[1]) != len(self.output_layer)
            ):
                raise IntegrityError("Incorrect output dataset entry!")
        # teach until the desired error is reached
        network_error = inf
        while network_error > precision_goal:
            network_error = self.teach(dataset)

    def SP_crossover(self, other: "Network") -> Tuple["Network", "Network"]:
        cls = type(self)
        n1 = self.export_data()
        n2 = other.export_data()
        assert len(n1) == len(n2)
        n = random.randint(1, len(n1) - 1)
        args = ([len(l) for l in self.layers], self.learning_rate, self.momentum_mod)
        n3, n4 = cls(*args), cls(*args)
        n3.import_data(n1[:n] + n2[n:])
        n4.import_data(n2[:n] + n1[n:])
        return (n3, n4)

    def R_crossover(self, other: "Network") -> Tuple["Network", "Network"]:
        cls = type(self)
        n1 = self.export_data()
        n2 = other.export_data()
        assert len(n1) == len(n2)
        args = ([len(l) for l in self.layers], self.learning_rate, self.momentum_mod)
        n3, n4 = cls(*args), cls(*args)
        n3_data, n4_data = [], []
        for n1_gene, n2_gene in zip(n1, n2):
            if random.random() < 0.5:
                n3_data.append(n1_gene)
                n4_data.append(n2_gene)
            else:
                n3_data.append(n2_gene)
                n4_data.append(n1_gene)
        n3.import_data(n3_data)
        n4.import_data(n4_data)
        return (n3, n4)

    def mutate(self):
        for l in self.layers:
            for n in l.neurons:
                for d in n.dendrons:
                    if random.random() < 0.01:
                        d.weight += d.weight * (random.random() * 0.2 - 0.1)

    def export_data(self) -> List[float]:
        data = []
        for layer in self.layers:
            data.extend(layer.export_data())
        return data

    def import_data(self, data: Union[Generator[float, None, None], List[float]]):
        with suppress(TypeError):
            if len(data) != sum(  # type: ignore
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
    __slots__ = ("network", "neurons")

    def __init__(self, network, neuron_count: int, previous_layer: Optional["Layer"] = None):
        self.network = network
        self.neurons = [Neuron(self, previous_layer) for _ in range(neuron_count)]

    def __len__(self):
        return len(self.neurons)

    def set_activation_f(self, func):
        for neuron in self.neurons:
            neuron.set_activation_f(func)

    def set_inputs(self, inputs: list):
        if len(inputs) != len(self.neurons):
            raise IntegrityError("Incorrect number of inputs!")
        for output, neuron in zip(inputs, self.neurons):
            neuron.set_output(output)

    def set_errors(self, targets: list):
        if len(targets) != len(self.neurons):
            raise IntegrityError("Incorrect number of outputs!")
        outputs = self.get_outputs()
        errors = [target - output for target, output in zip(targets, outputs)]
        for error, neuron in zip(errors, self.neurons):
            neuron.set_error(error)
        return sum(e ** 2 for e in errors)

    def get_outputs(self):
        return [neuron.output for neuron in self.neurons]

    def feed_forward(self):
        for neuron in self.neurons:
            neuron.feed_forward()

    def back_propagate(self):
        for neuron in self.neurons:
            neuron.back_propagate()

    def export_data(self) -> List[float]:
        data = []
        for neuron in self.neurons:
            data.extend(neuron.export_data())
        return data

    def import_data(self, data_iter: Iterator[float]):
        for neuron in self.neurons:
            neuron.import_data(data_iter)


class Neuron:
    __slots__ = ("layer", "dendrons", "error", "output", "bias", "activation_f")

    def __init__(self, layer, previous_layer: Optional[Layer] = None):
        self.layer = layer
        self.dendrons = []
        self.error = 0
        self.output = 0
        self.bias = 2 * random.random() - 1
        self.activation_f = tanh

        if previous_layer is not None:
            for neuron in previous_layer.neurons:
                con = Connection(neuron)
                self.dendrons.append(con)

    def __repr__(self):
        return "{}({}, {}, {})".format(self.__class__.__name__, self.output, self.bias, self.error)

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
        learning_rate = self.layer.network.learning_rate
        for dendron in self.dendrons:
            dendron.adjust_weight(gradient)
        self.bias += learning_rate * gradient
        self.error = 0

    def export_data(self) -> List[float]:
        # start with the bias
        data = [self.bias]
        # add the weights
        data.extend(d.weight for d in self.dendrons)
        return data

    def import_data(self, data_iter: Iterator[float]):
        # the first value is the bias
        self.bias = next(data_iter)
        # set weights
        for dendron in self.dendrons:
            dendron.weight = next(data_iter)


class Connection:
    __slots__ = ("source_neuron", "weight", "delta_weight")

    def __init__(self, source_neuron: Neuron):
        self.source_neuron = source_neuron
        self.weight = 2 * random.random() - 1
        self.delta_weight = 0

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.weight, self.delta_weight)

    def adjust_weight(self, gradient):
        network = self.source_neuron.layer.network
        delta = network.learning_rate * self.source_neuron.output * gradient
        self.delta_weight = (delta + network.momentum_mod * self.delta_weight)
        self.weight += self.delta_weight
        # pass the gradient to the source neuron
        self.source_neuron.add_error(self.weight * gradient)

    def get_value(self):
        return self.source_neuron.output * self.weight


# init the snake game
game = Game()
# run it faaaaaaaast
game.fps = 600


def dist(v1, v2):
    value = v1 - v2
    if value > 100:
        return 100
    if value < 0:
        return 0
    return value


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
        print(f"{i:3}: {net.fitness}, {net.strikes}")


def select_networks(nets):
    return networks[:len(networks) // 2]


def crossover_networks(nets):
    for n1, n2 in zip(nets[:-1:2], nets[1::2]):  # chunk by two
        nets.extend(n1.SP_crossover(n2))
    return nets


# create the population
population = 4
net_args = [12, 10, 8, 4]
networks = [Network(net_args) for _ in range(population)]

best_net = None
generation = 0
max_generation = 20
while True:
    generation += 1
    print(f"Generation: #{generation}")
    # evaluate for fitness
    evaluate_networks(networks)
    # sort with fittest at the top
    networks.sort(key=lambda n: n.fitness, reverse=True)
    print(f"Best fitness: {networks[0].fitness}")
    # save the best net
    if best_net is None or networks[0].fitness > best_net.fitness:
        best_net = networks[0]
        print(f"New best: {best_net.fitness}\n")
    if generation >= max_generation:
        break
    # select for the next generation
    networks = select_networks(networks)
    # crossover for the next generation
    networks = crossover_networks(networks)

# run the best saved network
print(f"\nRunning best network: {best_net.fitness} fitness")
# attach the controller
game.external = partial(controller, best_net)
game.fps = 8
while not game.window.has_exit:
    # reset the game
    game.reset()
    # run the game
    game.run()
    print(f"Score: {game.score}")
    sleep(1)
