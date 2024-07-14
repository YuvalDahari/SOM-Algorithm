\documentclass{article}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{graphicx}

\title{SOM Algorithm Implementation for Neuron Representation}
\author{}
\date{}

\begin{document}

\maketitle

\tableofcontents
\newpage

\section{Introduction}
This report describes the implementation of an algorithm for Self-Organizing Map (SOM) for classifying groups in data (in this case, numerical vectors) into representative neurons.

\section{Neuron Network}

\subsection{Neuron Representation}
When implementing an SOM neural network, the neurons are maintained in a data structure that updates during the algorithm's execution. The data structure chosen is a network made of hexagons, with each element being a vector of size 784, representing a pixel in a 28x28 grayscale image, where each cell in the vector neuron has a value between 0-255.

\subsection{Neuron Initialization}
We initialize each neuron differently to capture common data features effectively. This initialization involves iterating through the dataset, summing vectors element-wise for samples, and calculating the average to create the neuron vectors.

\subsection{Number of Neurons}
The network contains 100 neurons, a logical number given our data. This configuration ensures approximately ten neurons represent each digit, allowing for better differentiation between different variants.

\section{Best Matching Unit (BMU)}
In each iteration of the algorithm, we find the BMU for each data vector and update it and its neighbors. The BMU is the neuron vector from the network that is closest to the sampled data vector.

The Euclidean distance is used to determine the BMU:

\[
BMU = \arg\min_j \| \mathbf{x} - \mathbf{w}_j \|
\]

where:
\begin{itemize}
    \item $\mathbf{x}$ is the input vector,
    \item $\mathbf{w}_j$ is the weight vector of neuron $j$.
\end{itemize}

\section{Neighbor Update}
When updating a neuron, we also update its neighbors to ensure that similar neurons are close to each other. This process involves careful computation to avoid disrupting the entire network and gradually adjusting neighbors to the data vector.

The update rule for the weight vector of the BMU and its neighbors is:

\[
\mathbf{w}_j(t+1) = \mathbf{w}_j(t) + \eta(t) \cdot h_{j,i}(t) \cdot (\mathbf{x}(t) - \mathbf{w}_j(t))
\]

where:
\begin{itemize}
    \item $\mathbf{w}_j(t)$ is the weight vector of neuron $j$ at time $t$,
    \item $\eta(t)$ is the learning rate at time $t$,
    \item $h_{j,i}(t)$ is the neighborhood function centered around the BMU $i$.
\end{itemize}

The neighborhood function is often chosen as a Gaussian:

\[
h_{j,i}(t) = \exp\left(-\frac{\| \mathbf{r}_j - \mathbf{r}_i \|^2}{2\sigma^2(t)}\right)
\]

where:
\begin{itemize}
    \item $\mathbf{r}_j$ and $\mathbf{r}_i$ are the position vectors of neurons $j$ and $i$ in the grid,
    \item $\sigma(t)$ is the neighborhood radius at time $t$.
\end{itemize}

\section{Learning Rate}
The learning rate is a crucial value that influences the learning level at each stage of network building. It changes from iteration to iteration and differs between updating the BMU and its neighbors. Initially set at 0.3, the learning rate is reduced over iterations to retain learned information and prevent excessive changes.

\[
\eta(t) = \eta_0 \cdot \left(1 - \frac{t}{T}\right)
\]

where:
\begin{itemize}
    \item $\eta(t)$ is the learning rate at iteration $t$,
    \item $\eta_0$ is the initial learning rate,
    \item $T$ is the total number of iterations.
\end{itemize}

\section{Solution Quality}
To assess the quality of the solution, we can measure how well the network classifies new vectors. This involves evaluating the network's performance on a separate validation dataset and computing metrics such as accuracy, precision, recall, and F1 score.

\section{Hyperparameters}
The following hyperparameters are used in the SOM model:
\begin{itemize}
    \item Initial learning rate ($\eta_0$): 0.3
    \item Total number of iterations ($T$): 1000
    \item Neighborhood radius ($\sigma(t)$): decreasing over time
    \item Number of neurons: 100 (10x10 grid)
\end{itemize}

These hyperparameters can be adjusted to optimize the performance of the SOM algorithm for different datasets.

\section{Running Instructions}
To set up and run the project locally, follow these steps:

\subsection{Prerequisites}
\begin{itemize}
    \item Python 3.x
    \item NumPy
    \item Matplotlib (optional, for visualizing the results)
\end{itemize}

\subsection{Installation}
1. Clone the repository:
\begin{verbatim}
git clone https://github.com/yourusername/yourproject.git
\end{verbatim}
2. Navigate to the project directory:
\begin{verbatim}
cd yourproject
\end{verbatim}
3. Install the required Python packages:
\begin{verbatim}
pip install numpy matplotlib
\end{verbatim}

\subsection{Usage}
1. Prepare your dataset and ensure it is in the appropriate format (e.g., a CSV file with numerical vectors).
2. Run the SOM algorithm:
\begin{verbatim}
python som.py --input your_dataset.csv --output output_file
\end{verbatim}
3. (Optional) Visualize the results using Matplotlib:
\begin{verbatim}
python visualize.py --input output_file
\end{verbatim}

\section{Contributing}
We welcome contributions to this project. If you would like to contribute, please follow these guidelines:
\begin{itemize}
    \item Fork the repository
    \item Create a new branch (\texttt{git checkout -b feature-branch})
    \item Make your changes and commit them (\texttt{git commit -m 'Add new feature'})
    \item Push to the branch (\texttt{git push origin feature-branch})
    \item Open a pull request
\end{itemize}

\section{License}
This project is licensed under the MIT License. See the \texttt{LICENSE} file for details.

\end{document}
