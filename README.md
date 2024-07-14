# SOM Algorithm Implementation for Neuron Representation

This repository contains the implementation of a Self-Organizing Map (SOM) algorithm for classifying groups in data (numerical vectors) into representative neurons.

## Neuron Network

### Neuron Representation

When implementing an SOM neural network, the neurons are maintained in a data structure that updates during the algorithm's execution. The data structure chosen is a network made of hexagons, with each element being a vector of size 784, representing a pixel in a 28x28 grayscale image, where each cell in the vector neuron has a value between 0-255.

### Neuron Initialization

We initialize each neuron differently to capture common data features effectively. This initialization involves iterating through the dataset, summing vectors element-wise for samples, and calculating the average to create the neuron vectors.

### Number of Neurons

The network contains 100 neurons, a logical number given our data. This configuration ensures approximately ten neurons represent each digit, allowing for better differentiation between different variants.

## Best Matching Unit (BMU)

In each iteration of the algorithm, we find the BMU for each data vector and update it and its neighbors. The BMU is the neuron vector from the network that is closest to the sampled data vector.

The Euclidean distance is used to determine the BMU:

\[ BMU = \arg\min_j \| \mathbf{x} - \mathbf{w}_j \| \]

where:
- \( \mathbf{x} \) is the input vector,
- \( \mathbf{w}_j \) is the weight vector of neuron \( j \).

## Neighbor Update

When updating a neuron, we also update its neighbors to ensure that similar neurons are close to each other. This process involves careful computation to avoid disrupting the entire network and gradually adjusting neighbors to the data vector.

The update rule for the weight vector of the BMU and its neighbors is:

\[ \mathbf{w}_j(t+1) = \mathbf{w}_j(t) + \eta(t) \cdot h_{j,i}(t) \cdot (\mathbf{x}(t) - \mathbf{w}_j(t)) \]

where:
- \( \mathbf{w}_j(t) \) is the weight vector of neuron \( j \) at time \( t \),
- \( \eta(t) \) is the learning rate at time \( t \),
- \( h_{j,i}(t) \) is the neighborhood function centered around the BMU \( i \).

The neighborhood function is often chosen as a Gaussian:

\[ h_{j,i}(t) = \exp\left(-\frac{\| \mathbf{r}_j - \mathbf{r}_i \|^2}{2\sigma^2(t)}\right) \]

where:
- \( \mathbf{r}_j \) and \( \mathbf{r}_i \) are the position vectors of neurons \( j \) and \( i \) in the grid,
- \( \sigma(t) \) is the neighborhood radius at time \( t \).

## Learning Rate

The learning rate is a crucial value that influences the learning level at each stage of network building. It changes from iteration to iteration and differs between updating the BMU and its neighbors. Initially set at 0.3, the learning rate is reduced over iterations to retain learned information and prevent excessive changes.

\[ \eta(t) = \eta_0 \cdot \left(1 - \frac{t}{T}\right) \]

where:
- \( \eta(t) \) is the learning rate at iteration \( t \),
- \( \eta_0 \) is the initial learning rate,
- \( T \) is the total number of iterations.

## Solution Quality

To assess the quality of the solution, we can measure how well the network classifies new vectors. This involves evaluating the network's performance on a separate validation dataset and computing metrics such as accuracy, precision, recall, and F1 score.

## Hyperparameters

The following hyperparameters are used in the SOM model:
- Initial learning rate (\( \eta_0 \)): 0.3
- Total number of iterations (\( T \)): 1000
- Neighborhood radius (\( \sigma(t) \)): decreasing over time
- Number of neurons: 100 (10x10 grid)

These hyperparameters can be adjusted to optimize the performance of the SOM algorithm for different datasets.

## Running Instructions

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib (optional, for visualizing the results)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YuvalDahari/SOM-Algorithm.git
   ```
2. Install the required Python packages:
      ```bash
    pip install numpy matplotlib
   ```
### Usage

1. Prepare your dataset and ensure it is in the appropriate format (e.g., a CSV file with numerical vectors).
2. Run the SOM algorithm:
      ```bash
    python main.py
   ```
