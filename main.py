import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import ball
import os

# Import the custom image utility functions
from image_utils import load_image, edge_detection

def main():
    # Define the path to the original image
    # IMPORTANT: Replace '134084821959131087.jpg' with the actual path to your image file
    # If the file doesn't exist, a dummy image will be created for demonstration.
    image_path = '134084821959131087.jpg'
    output_image_name = 'my_edges.png'

    # Create a dummy image for testing if the specified image_path does not exist.
    if not os.path.exists(image_path):
        print(f"Image file not found at '{image_path}'. Creating a dummy image for demonstration.")
        dummy_img_array = np.random.randint(0, 256, size=(100, 150, 3), dtype=np.uint8)
        dummy_img = Image.fromarray(dummy_img_array)
        dummy_img.save(image_path)
        print(f"Dummy image saved as '{image_path}'.")

    print(f"Loading image from: {image_path}")
    original_image = load_image(image_path)

    if original_image is None:
        print("Exiting due to image loading failure.")
        return
