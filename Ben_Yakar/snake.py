import streamlit as st
import cv2
import numpy as np
from skimage import io, color
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from PIL import Image

def find_contours(image, snake_points, alpha=0.015, beta=10, gamma=0.001):
    # Apply Gaussian smoothing to the image
    img_smooth = gaussian(image, 3)

    # Apply the active contour model (snakes)
    snake = active_contour(img_smooth, snake_points, alpha=alpha, beta=beta, gamma=gamma)

    return snake

def normalize_image(image):
    # Clip and normalize the image to the range [0, 255]
    # image = np.clip(image, 0, 255)
    # image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20,20))
    contrast_enhanced = clahe.apply(image)
    return contrast_enhanced


def main():
    st.title("Active Contour Model (Snakes) for Contour Detection")
    st.write("Upload a PNG image and the app will find contours using the active contour model.")

    # Upload image
    uploaded_file = st.file_uploader("Choose a PNG image", type="png")
    
    if uploaded_file is not None:
        # Read the uploaded image
        image = Image.open(uploaded_file)
        # Get image mode
        image_mode = image.mode
        # Convert image if necessary
        if image_mode == 'I':
            # Convert to NumPy array
            image_array = np.array(image)
            # Normalize the image data
            image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
            # Convert back to PIL image
            image = Image.fromarray(image_array)
            # Convert to grayscale ('L') for display purposes
            image = image.convert('L')
         # Display the image
        
        st.image(image, caption='Uploaded Image', use_column_width=True)
        

        image = np.array(image)

        # Convert the image to grayscale if it's not already
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image

        # Normalize the image data to [0, 255]
        normalized_image = normalize_image(gray_image)

        # Display the uploaded image
        st.image(normalized_image, caption="Uploaded Image", use_column_width=True)

        # Create an initial snake starting points (for demo purposes, this can be a circle around the center)
        s = np.linspace(0, 2 * np.pi, 400)
        r = gray_image.shape[0] // 2 + 100 * np.sin(s)
        c = gray_image.shape[1] // 2 + 100 * np.cos(s)
        init = np.array([r, c]).T

        # # Find the contours using active contour model (snakes)
        # snake = find_contours(gray_image, init)

        # # Create a blank image to draw the contour on
        # contour_image = np.copy(normalized_image)
        # for point in snake:
        #     rr, cc = point
        #     contour_image[int(rr), int(cc)] = 255  # Draw contour in white

        # # Display the result
        # st.image(contour_image, caption="Contour Detection Result", use_column_width=True)

if __name__ == "__main__":
    main()
