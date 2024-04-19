import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import pickle
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

# Load precomputed embeddings and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load pre-trained ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Define model for feature extraction
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

# Initialize Nearest Neighbors
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(feature_list)


# Function to compute indices of similar images
def compute_indices(input_image):
    img = image.load_img(input_image, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    distances, indices = neighbors.kneighbors([normalized_result])
    return indices


# Function to display similar images
def display_similar_images(indices):
    for i, file_index in enumerate(indices[0][1:6]):
        temp_img = cv2.imread(filenames[file_index])
        temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        temp_img = Image.fromarray(temp_img)
        temp_img.thumbnail((512, 512))
        temp_img = ImageTk.PhotoImage(temp_img)

        panel = tk.Label(root, image=temp_img)
        panel.image = temp_img
        panel.grid(row=1, column=i + 1, padx=10, pady=10)


# Function to handle button click event
def find_similar_images():
    file_path = filedialog.askopenfilename()
    if file_path:
        indices = compute_indices(file_path)
        display_similar_images(indices)


# Create the Tkinter window
root = tk.Tk()
root.title("Similar Image Finder")


# Button to open file dialog and display selected image
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img.thumbnail((300, 300))  # Resize image to fit in the GUI
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(root, image=img)
        panel.image = img
        panel.grid(row=1, column=0, padx=10, pady=10)


# Button to open file dialog
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.grid(row=0, column=0, padx=10, pady=10)

# Button to find and display similar images
find_similar_button = tk.Button(root, text="Find Similar", command=find_similar_images)
find_similar_button.grid(row=0, column=1, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
