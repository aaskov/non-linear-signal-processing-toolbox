# -*- coding: utf-8 -*-
"""
Neural network - nsp
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt


def draw_circle(position, neuron_radius):
    circle = plt.Circle(position, radius=neuron_radius, fill=True, fc='Black')
    plt.gca().add_patch(circle)


def draw_line(position1, position2, weight=0.8):
    if weight >= 0:
        #Positive
        line = plt.Line2D((position1[0], position2[0]),
                          (position1[1], position2[1]),
                          color='Green',
                          linewidth=weight)
    else:
        # Negative
        line = plt.Line2D((position1[0], position2[0]),
                          (position1[1], position2[1]),
                          color='Red',
                          linewidth=abs(weight))

    plt.gca().add_line(line)


def draw_network(layer, Wi, Wo, title='Neural network'):
    neuron_distance = 3
    neuron_radius = 0.2
    layer_distance = 4
    padding = 1.5

    network_height = max(layer) + 1  # add 1 for bias neuron
    network_width = len(layer)

    center = (network_height-1)*neuron_distance/2

    # Setup the figure
    xmin = -padding
    xmax = (network_width-1)*layer_distance + padding
    ymin = -padding
    ymax = (network_height-1)*neuron_distance + padding
    fig = plt.figure('Diagram')
    fig.suptitle(title)
    plt.axis('scaled')
    ax = fig.add_subplot(111)
    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_xlabel('Layers')
    plt.xticks(np.arange(network_width)*layer_distance,
               np.arange(network_width))

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

    # Draw connection (2)
    for level in range(len(neurons[:-1])):
        ii = 0
        for i in np.arange(len(neurons[level]), 0, -1)-1:
            jj = 0
            for j in np.arange(len(neurons[level+1]), 0, -1)-1:
                # Take top left first, connect its neurons in level+1, continue
                # Skip the bias neuron
                if j != 0:
                    weight = Wi[ii, jj]
                    draw_line(neurons[level][i], neurons[level+1][j], weight)
                    jj = jj + 1

                if (level+1) == (network_width-1):
                    # Output layer
                    weight = Wo[ii]
                    draw_line(neurons[level][i], neurons[level+1][j], weight)
                    jj = jj + 1

            ii = ii + 1

    # Draw neurons
    for level in neurons:
        for neuron in level:
            if neuron == level[0]:
                # Bias
                draw_circle(neuron, neuron_radius)
                if level is not neurons[network_width-1]:
                    ax.text(neuron[0]-0.5, neuron[1]-1.0, r'Bias', fontsize=10)
            else:
                draw_circle(neuron, neuron_radius)

    # Finish
    plt.draw()
    plt.show()

if __name__ == "__main__":
    # Define the layers (without bias, and no output bias)
    layer = [2, 8, 1]

    multiply = 2.0
    Wi = np.random.randn(2+1, 8)*multiply
    Wo = np.random.randn(8+1, 1)*multiply

    print Wi
    print Wo

    draw_network(layer, Wi, Wo)

