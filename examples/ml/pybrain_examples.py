from pybrain.tools.shortcuts import buildNetwork

#    create a network with two inputs, three hidden
#    and a single output neuron.
net = buildNetwork(2, 3, 1)
net.activate([2, 1])

