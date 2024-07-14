import pandas as pd
import time
import tkinter as tk
import math
import numpy as np
from PIL import Image, ImageTk

# Files' constants
LABELS = []
DS = []
DS_PATH = './digits_test.csv'  # Path to the dataset CSV file
LABELS_PATH = './digits_keys.csv'  # Path to the labels CSV file

# Algorithm's constants
MATRIX_SIZE = 10  # Size of the neuron matrix (10x10)
NEURONS_MATRIX = []  # Matrix to store neuron vectors
ITERATIONS = 1  # Number of iterations
LEARNING_RATE = 0.3  # Learning rate for updating neurons
MAX_ITERATIONS = 5  # Maximum number of iterations to run
MAX_RUN_TIME = 60  # Maximum runtime in seconds (2.5 minutes)
VECTOR_LENGTH = 784  # Size of the vectors
HEX_SIZE = 40  # Size if the hexagon


# Class for create hexagons board
class HexBoard(tk.Canvas):
    # Function to init start parameters board
    def __init__(self, master=None, rows=MATRIX_SIZE, cols=MATRIX_SIZE, hex_size=HEX_SIZE, **kwargs):
        super().__init__(master, **kwargs)
        self.hex_size = hex_size
        self.hex_width = self.hex_size * 2
        self.hex_height = math.sqrt(3) * self.hex_size
        self.hex_horizontal_distance = self.hex_width * 3 / 4
        self.hex_vertical_distance = self.hex_height
        self.colors = [
            "#FF5733",  # Bright Red
            "#33FF57",  # Bright Green
            "#3357FF",  # Bright Blue
            "#F1C40F",  # Bright Yellow
            "#9B59B6",  # Bright Purple
            "#E67E22",  # Bright Orange
            "#1ABC9C",  # Bright Teal
            "#2ECC71",  # Bright Mint
            "#3498DB",  # Bright Sky Blue
            "#E74C3C"  # Bright Coral
        ]

        canvas_width = cols * self.hex_horizontal_distance + self.hex_size / 2
        canvas_height = rows * self.hex_vertical_distance + self.hex_size

        self.configure(width=canvas_width, height=canvas_height)
        self.hexagons = {}  # Dictionary to store hexagons and their center points
        self.images = {}  # Dictionary to store references to images
        self.create_hex_board(rows, cols)

    # Function to create the board
    def create_hex_board(self, rows, cols):
        for row in range(rows):
            for col in range(cols):
                x = col * self.hex_horizontal_distance + self.hex_size
                y = row * self.hex_vertical_distance + self.hex_size
                if col % 2 == 1:
                    y += self.hex_vertical_distance / 2
                hex_id, center = self.draw_hexagon(x, y)
                self.hexagons[(row, col)] = (hex_id, center)  # Store hexagon by row, col with center point

    # Function to create the hexagons in the board
    def draw_hexagon(self, x_center, y_center):
        points = []
        for i in range(6):
            angle = math.radians(60 * i)
            x = x_center + self.hex_size * math.cos(angle)
            y = y_center + self.hex_size * math.sin(angle)
            points.append((x, y))
        hex_id = self.create_polygon(points, outline='white', fill='black', width=2)
        return hex_id, (x_center, y_center)

    # Function add image (vector) to hexagon in place (row,col)
    def add_vector_as_image(self, row, col, vector):
        x_center, y_center = self.hexagons[(row, col)][1]
        # Convert vector to grayscale image
        size = int(math.sqrt(len(vector)))  # Assuming vector length is a perfect square
        img = Image.fromarray(np.uint8(vector.reshape((size, size))), 'L')
        img = img.resize((int(self.hex_size * 1.3), int(self.hex_size * 1.3)), Image.LANCZOS)
        photo_img = ImageTk.PhotoImage(img)

        # Create an image on the canvas
        self.create_image(x_center, y_center, image=photo_img)

        # Store reference to prevent garbage collection
        self.images[(x_center, y_center)] = photo_img

    # Function return distance between two hexagons
    def get_distance_between_two_hexagon(self, idx1, idx2):
        x1, y1 = self.hexagons[idx1][1]
        x2, y2 = self.hexagons[idx2][1]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Function add text to hexagon in place (row,col)
    def add_text_to_hex(self, row, col, number_text, percentage_text):
        x_center, y_center = self.hexagons[(row, col)][1]
        color = self.colors[int(number_text)]
        self.create_text(x_center, y_center - 10, text=number_text, fill=color)
        self.create_text(x_center, y_center + 10, text=percentage_text, fill=color)

    # Function return topological distance between two hexagons
    def topological_distance(self, idx1, idx2):
        counter = 0
        current_hex = idx1
        while idx2 != current_hex:
            counter += 1
            min_point = (0, 0)
            min_distance = 10000000000
            if current_hex[1] % 2 == 1:
                first_layer_directions = [(1, -1), (1, 1), (-1, 0), (0, -1), (0, 1), (1, 0)]
            else:
                first_layer_directions = [(-1, -1), (-1, 1), (-1, 0), (0, 1), (0, -1), (1, 0)]
            for di, dj in first_layer_directions:
                ni, nj = current_hex[0] + di, current_hex[1] + dj
                if 0 <= ni < 10 and 0 <= nj < 10:
                    distance = self.get_distance_between_two_hexagon(idx2, (ni, nj))
                    if distance < min_distance:
                        min_point = (ni, nj)
                        min_distance = distance
            current_hex = min_point
        return counter


# Read data set
def read_data_set():
    global DS
    global VECTOR_LENGTH

    data_file = pd.read_csv(DS_PATH)
    DS = [np.array(row) for row in data_file.values]  # List of the data as numpy arrays
    VECTOR_LENGTH = len(DS[0])


# Read labels
def read_true_labels():
    global LABELS
    df = pd.read_csv(LABELS_PATH)
    LABELS = df.values.tolist()


# Function to create averaged 10x10 matrix from DS vectors
def init_matrix():
    global NEURONS_MATRIX

    # Initialize a matrix of lists to store vectors
    matrix = [[[] for _ in range(MATRIX_SIZE)] for _ in range(MATRIX_SIZE)]

    # Iterate over DS and distribute vectors into the matrix
    for idx, vector in enumerate(DS):
        row = (idx // MATRIX_SIZE) % MATRIX_SIZE
        col = idx % MATRIX_SIZE
        matrix[row][col].append(vector)

    # Calculate the average for each cell
    NEURONS_MATRIX = np.zeros((MATRIX_SIZE, MATRIX_SIZE, VECTOR_LENGTH))
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            if matrix[i][j]:  # Avoid division by zero
                NEURONS_MATRIX[i][j] = np.mean(matrix[i][j], axis=0)


# Function to calculate Euclidean distance between two vectors
def euclidean_distance(vector1, vector2):
    # Convert inputs to numpy arrays (if they aren't already)
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Calculate the Euclidean distance
    distance = np.sqrt(np.sum((vector1 - vector2) ** 2))

    return distance


# Function to find the Best Matching Unit (BMU) index for a given vector
def extract_bmu_idx(vector):
    min_distance = float('inf')
    closest_neuron_idx = (0, 0)

    # Find the closest neuron
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            distance = euclidean_distance(vector, NEURONS_MATRIX[i][j])
            if distance < min_distance:
                min_distance = distance
                closest_neuron_idx = (i, j)

    return closest_neuron_idx


# Function to calculate the update for a neuron based on a vector and learning rate
def calc_update(vector, neurons, learning_rate):
    return neurons + learning_rate * (vector - neurons)


# Function to update the neighbors of the BMU
def neighbors_update(vector, idx):
    global NEURONS_MATRIX

    first_learning_rate = LEARNING_RATE / (ITERATIONS + 1)
    second_learning_rate = first_learning_rate / (ITERATIONS + 1)
    i, j = idx

    first_layer = []
    second_layer = []
    first_layer_directions = []
    second_layer_directions = []

    if j % 2 == 1:
        # Define the directions for the first layer neighbors
        first_layer_directions = [(0, -1), (0, 1), (1, -1), (1, 0), (1, 1), (-1, 0)]
        # Define the directions for the second layer neighbors
        second_layer_directions = [(-2, 0), (-1, 1), (-1, 2), (0, 2), (1, 2), (2, 1),
                                   (2, 0), (-1, -1), (-1, -2), (0, -2), (1, -2), (2, -1)]

    if j % 2 == 0:
        # Define the directions for the first layer neighbors
        first_layer_directions = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (0, -1), (1, 0)]
        # Define the directions for the second layer neighbors
        second_layer_directions = [(-2, -1), (-2, 0), (-2, 1), (-1, 2), (0, 2), (1, 2),
                                   (-1, -2), (0, -2), (1, -2), (1, -1), (1, 1), (2, 0)]

    for di, dj in first_layer_directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < 10 and 0 <= nj < 10:
            first_layer.append((ni, nj))

    for di, dj in second_layer_directions:
        ni, nj = i + di, j + dj
        if 0 <= ni < 10 and 0 <= nj < 10:
            second_layer.append((ni, nj))

    # Update first layer neighbors
    for ni, nj in first_layer:
        NEURONS_MATRIX[ni][nj] = calc_update(vector, NEURONS_MATRIX[ni][nj], first_learning_rate)

    # Update second layer neighbors
    for ni, nj in second_layer:
        NEURONS_MATRIX[ni][nj] = calc_update(vector, NEURONS_MATRIX[ni][nj], second_learning_rate)


# Function to update the neurons using the dataset
def update():
    global NEURONS_MATRIX

    for vector in DS:
        idx = extract_bmu_idx(vector)
        i, j = idx
        NEURONS_MATRIX[i][j] = calc_update(vector, NEURONS_MATRIX[i][j], LEARNING_RATE)
        neighbors_update(vector, idx)


#  Find the location of two best neurons according to the vector
def find_two_best_neurons(vector):
    first_min_distance = float('inf')
    second_min_distance = float('inf')
    first_closest_neuron_idx = (0, 0)
    second_closest_neuron_idx = (0, 0)

    # Find the closest neuron
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            distance = euclidean_distance(vector, NEURONS_MATRIX[i][j])
            if distance < first_min_distance:
                second_min_distance = first_min_distance
                second_closest_neuron_idx = first_closest_neuron_idx
                first_min_distance = distance
                first_closest_neuron_idx = (i, j)
            elif distance < second_min_distance:
                second_min_distance = distance
                second_closest_neuron_idx = (i, j)

    return first_closest_neuron_idx, second_closest_neuron_idx


# Function to calculate a score
def calc_score(board):
    total_score = 0
    num_sample = 0
    max_euclidean_distance_value = np.sqrt(VECTOR_LENGTH * (255 ** 2))
    max_topological_distance_value = 13

    for vector in DS:
        num_sample = num_sample + 1
        i, j = extract_bmu_idx(vector)
        error_euclidean_distance = 100 * (euclidean_distance(vector,
                                                             NEURONS_MATRIX[i][j]) / max_euclidean_distance_value)

        first_closest_neuron_idx, second_closest_neuron_idx = find_two_best_neurons(vector)
        topological_distance = board.topological_distance(first_closest_neuron_idx, second_closest_neuron_idx) - 1
        error_topological_distance = 100 * (topological_distance / max_topological_distance_value)

        total_score = total_score + (100 - (0.5 * error_euclidean_distance + 0.5 * error_topological_distance))

    return total_score / num_sample


# Function to perform matching and create a matching matrix
def matching():
    matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE, 10))  # Initialize matrix with zeros

    # Populate matrix with vectors from DS and LABELS
    for idx, vector in enumerate(DS):
        i, j = extract_bmu_idx(vector)
        label = LABELS[idx]
        matrix[i][j][label] += 1

    dtype = [('max_idx', int), ('percentage', float)]
    matching_matrix = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=dtype)

    # Populate matching_matrix based on matrix
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            vector = matrix[i][j]
            total_sum = np.sum(vector)
            max_idx = np.argmax(vector)
            percentage = (vector[max_idx] / total_sum) * 100
            matching_matrix[i][j] = (max_idx, percentage)

    return matching_matrix


# Function to draw the SOM grid
def draw(matching_matrix):
    root = tk.Tk()
    root.title("SOM Board")
    frame1 = tk.Frame(root)
    frame1.grid(row=0, column=0)
    frame2 = tk.Frame(root)
    frame2.grid(row=0, column=1)
    board = HexBoard(frame1, rows=MATRIX_SIZE, cols=MATRIX_SIZE, hex_size=HEX_SIZE)
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            vector = NEURONS_MATRIX[i][j]
            board.add_vector_as_image(i, j, vector)

    board.pack()
    score_text = 'The SOM score: ' + str(format(calc_score(board), '.2f'))
    text_label = tk.Label(frame1, text=score_text, font=('Helvetica', 16, 'bold'))
    text_label.pack()
    board2 = HexBoard(frame2, rows=MATRIX_SIZE, cols=MATRIX_SIZE, hex_size=HEX_SIZE)
    for i in range(MATRIX_SIZE):
        for j in range(MATRIX_SIZE):
            number, percentage = matching_matrix[i][j]
            board2.add_text_to_hex(i, j, str(number), f'{percentage:.2f}%')
    board2.pack()
    root.mainloop()


# Main function to run the SOM algorithm
def main():
    global LEARNING_RATE
    global ITERATIONS

    start_time = time.time()
    read_data_set()  # Read the dataset
    init_matrix()  # Initialize the neuron matrix

    for i in range(MAX_ITERATIONS):
        update()  # Update the neurons
        LEARNING_RATE /= 2
        if time.time() - start_time > MAX_RUN_TIME:
            break
        ITERATIONS = ITERATIONS + 1

    print(time.time() - start_time)
    read_true_labels()  # Read the true labels
    matching_matrix = matching()  # Perform matching and get the matching matrix
    draw(matching_matrix)  # Drawing


if __name__ == '__main__':
    main()
