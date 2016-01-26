from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def draw_circle(position, neuron_radius):
    circle = plt.Circle(position, radius=neuron_radius, fill=True)
    plt.gca().add_patch(circle)


def draw_line(position1, position2):
    line = plt.Line2D((position1[0], position2[0]),
                      (position1[1], position2[1]))
    plt.gca().add_line(line)


def draw_network(layer, title='Neural network'):
    neuron_distance = 3
    neuron_radius = 0.2
    layer_distance = 4
    padding = 1.0

    network_height = max(layer) + 1  # add 1 for bias neuron
    network_width = len(layer)

    center = (network_height-1)*neuron_distance/2

    # Setup the figure
    xmin = -padding
    xmax = (network_width-1)*layer_distance + padding
    ymin = -padding
    ymax = (network_height-1)*neuron_distance + padding
    plt.figure()
    plt.axis('scaled')
    plt.title(title)
    plt.axis([xmin, xmax, ymin, ymax])

    neurons = []

    # Get positions for all neurons
    for level in range(network_width):
        if level < (network_width-1):
            n = layer[level] + 1
        else:
            n = layer[level]

        displacement = np.linspace(-(n-1)/2, (n-1)/2, n)*neuron_distance
        position_y = np.ones(n)*center + displacement

        neuron_set = []

        for position in position_y:
            position_x = layer_distance*level
            neuron_set.append((position_x, position))

        neurons.append(neuron_set)

    # Draw connections
    for level in range(len(neurons[:-1])):
        for neuron in neurons[level]:
            for connection in neurons[level+1]:
                if connection is not neurons[level+1][0]:
                    draw_line(neuron, connection)
                if (level+1) is (network_width-1):
                    draw_line(neuron, connection)

    # Draw neurons
    for level in neurons:
        for neuron in level:
            draw_circle(neuron, neuron_radius)

    # Finish
    plt.draw()
    plt.show()

if __name__ is "__main__":

    # Define the layers (without bias, and no output bias)
    layer = [2, 3, 3, 2]

    draw_network(layer)
