from neural_network import Network

# define
net = Network([2, 3, 3])
dataset = [
    ((0, 0), (0, 0, 0)),
    ((0, 1), (0, 1, 1)),
    ((1, 0), (0, 1, 1)),
    ((1, 1), (1, 1, 0)),
]
# teach
for net_error in net.teach_loop(dataset):
    print(net_error)
# evaluate results
while True:
    values = input("Inputs: ")
    while True:
        try:
            inputs = tuple(int(v) for v in values.split())
        except ValueError:
            pass
        else:
            break
    # eval
    outputs = net.feed_forward(inputs)
    print(outputs)
    print([round(o) for o in outputs])
