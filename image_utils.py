from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(file_path):
    """
    Loads a color image from the given file path and converts it into a NumPy array.

    Args:
        file_path (str): The path to the image file.

    Returns:
        np.array: The image as a NumPy array.
    """
    try:
        # Open the image file
        img = Image.open(file_path)
        # Convert the image to a NumPy array
        img_array = np.array(img)
        return img_array
    except FileNotFoundError:
        print(f"Error: Image file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the image: {e}")
        return None

def edge_detection(image_array):
    """
    Performs edge detection on an image array.

    Args:
        image_array (np.array): The input image as a NumPy array (3-channel color).

    Returns:
        np.array: The edge magnitude array.
    """
    # 1. Convert to grayscale
    # Average the three color channels
    grayscale_image = np.mean(image_array, axis=2)

    # 2. Create kernelY filter
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    # 3. Create kernelX filter
    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    # 4. Apply filters using convolve2d with zero padding
    # mode='same' ensures output has same size as input
    # boundary='fill' with fillvalue=0 implements zero padding
    edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # 5. Compute edgeMAG
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
