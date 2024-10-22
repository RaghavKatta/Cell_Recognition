import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from sobel import sobel_edge_filter

def power_law_normalize(image_array, gamma=2):
    # Normalize the image array
    image_array = (image_array - np.min(image_array))  # Shift to start from 0
    image_array = (image_array / np.max(image_array))  # Scale to 0-1
    image_array = np.power(image_array, gamma)  # Apply power-law
    image_array = (image_array * 255).astype(np.uint8)  # Scale to 0-255
    return image_array

def calculate_sharpness(image, dark_threshold=30):
    # Convert image to NumPy array
    image_array = np.array(image)
    image_array2 = image_array.copy()
    # Convert to grayscale if not already
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)


    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#   # contrast_enhanced = clahe.apply(image_array)

    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(image_array, cv2.CV_64F)

    # Measure sharpness (absolute value of the Laplacian)
    sharpness = np.abs(laplacian)

    dark_pixels = image_array < dark_threshold
    # sharpness[dark_pixels] = 10  # Mark as sharp
    sharpness_normalized = power_law_normalize(sharpness, gamma=0.3)  # Adjust gamma as needed
    # sharpness_image = Image.fromarray(sharpness_normalized)

    # Initialize the final sharpness image with the normalized sharpness
    final_sharpness_image = np.copy(sharpness_normalized)

    # # Preserve original values for dark pixels
    # final_sharpness_image[dark_pixels] =  255 - image_array2[dark_pixels]

    # Convert the result to PIL format
    sharpness_image_pil = Image.fromarray(final_sharpness_image)

    return sharpness_image_pil


# def adaptive_sharpness_measure(image, intensity_weight=0.5, laplacian_threshold = 10):
#     # Convert to grayscale if not already
#     image_array = np.array(image)

#     if len(image_array.shape) == 3:
#         image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

#     # Apply the Laplacian operator for edge detection
#     laplacian = cv2.Laplacian(image_array, cv2.CV_64F)
#     laplacian_abs = np.abs(laplacian)
#     laplacian_norm = (laplacian_abs / np.max(laplacian_abs)) * 255

#     # # Local contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
#     # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#     # contrast_enhanced = clahe.apply(image_array)

#     # # Adaptive weighting based on local contrast
#     # adaptive_sharpness = laplacian_norm * (contrast_enhanced / 255.0) ** intensity_weight + laplacian_norm * (255.0/contrast_enhanced ) ** intensity_weight
    
    
#     # return adaptive_sharpness

#     # Apply thresholding to the Laplacian
#     _, thresholded_laplacian = cv2.threshold(laplacian_abs, laplacian_threshold, 255, cv2.THRESH_BINARY)

#     # Normalize the result to 8-bit image
#     thresholded_laplacian = np.uint8(thresholded_laplacian)
#     adaptive_sharpness = laplacian_norm * (thresholded_laplacian / 255.0) ** intensity_weight + laplacian_norm * (255.0/thresholded_laplacian) ** intensity_weight
    
    
#     return thresholded_laplacian


def laplacian_threshold_local(image, laplacian_threshold=50, window_size=100):
    # Convert to grayscale if the image is in color
    image_array = np.array(image)

    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    # Compute the Laplacian to detect edges
    laplacian = cv2.Laplacian(image_array, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)

    
    # Normalize the Laplacian values to be between 0 and 1
    laplacian_abs = cv2.normalize(laplacian_abs, None, 0, 1, cv2.NORM_MINMAX)

    # Calculate the local mean and standard deviation within the window
    local_mean = cv2.blur(laplacian_abs, (window_size, window_size))
    local_std = cv2.blur(laplacian_abs**2, (window_size, window_size)) - local_mean**2
    local_std = np.sqrt(np.maximum(local_std, 0))  # Ensure non-negative values

    # Clip the local mean and std to avoid extreme thresholds
    local_mean = np.clip(local_mean, 0, 1)
    local_std = np.clip(local_std, 0, 1)

    # Adaptive thresholding based on local statistics
    adaptive_threshold = local_mean + laplacian_threshold * local_std
    adaptive_threshold = .1
    # Thresholding: retain areas where Laplacian is higher than the adaptive threshold
    thresholded_laplacian = (laplacian_abs > adaptive_threshold).astype(np.uint8) * 255

    return thresholded_laplacian

    
#dark double validation

def invert_colors(image):
    # Convert to NumPy array
    image_array = np.array(image)
    
    # Invert colors
    inverted_image_array = 255 - image_array
    
    return Image.fromarray(inverted_image_array)

#Use of power law for easier adjustability over explicitly linear or logorithmic methods


def reduce_noise(image_array, method='gaussian', **kwargs):
    if method == 'gaussian':
        return gaussian_blur(image_array, **kwargs)
    elif method == 'median':
        return median_filter(image_array, **kwargs)
    elif method == 'bilateral':
        return bilateral_filter(image_array, **kwargs)
    elif method == 'nlmeans':
        return non_local_means_denoising(image_array, **kwargs)
    elif method == 'none':
        return image_array
    else:
        raise ValueError("Invalid noise reduction method")

def gaussian_blur(image_array, kernel_size=5):
    return cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)

def median_filter(image_array, kernel_size=5):
    return cv2.medianBlur(image_array, kernel_size)

def bilateral_filter(image_array, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image_array, d, sigma_color, sigma_space)

def non_local_means_denoising(image_array, h=10):
    return cv2.fastNlMeansDenoising(image_array, None, h, 7, 21)

def apply_otsu_threshold(image_array):
    _, thresholded = cv2.threshold(image_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def preprocess_image(image):
    # Convert the Pillow image to a NumPy array
    image_array = np.array(image)
    
    # Convert to grayscale if not already (assumes image is in RGB format)
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to the NumPy array
    kernel_size = 5  # Define kernel size or pass as parameter
    blurred_image_array = cv2.GaussianBlur(image_array, (kernel_size, kernel_size), 0)
    
    return blurred_image_array

def enhance_contrast(image, lower_percentile=1, upper_percentile=99):
    # Convert to grayscale if not already
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Calculate the percentiles
    lower_threshold = np.percentile(image_array, lower_percentile)
    upper_threshold = np.percentile(image_array, upper_percentile)
    
    # Clip the pixel values between the thresholds
    image_array_clipped = np.clip(image_array, lower_threshold, upper_threshold)
    
    # Normalize the image to the range [0, 255]
    image_normalized = (image_array_clipped - lower_threshold) / (upper_threshold - lower_threshold) * 255
    image_normalized = image_normalized.astype(np.uint8)
    
    # Convert back to PIL image
    enhanced_image = Image.fromarray(image_normalized)
    
    return enhanced_image


def apply_binary_threshold(image_array, threshold):
    binary_image_array = (image_array > threshold) * 255
    return binary_image_array.astype(np.uint8)

def main():
    st.title("BG REMOVAL")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a PNG file", type="png")

    if uploaded_file is not None:
        # Open the image
        image = Image.open(uploaded_file)
        # Get image mode
        image_mode = image.mode
        st.write(f"Image mode: {image_mode}")
        
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
        
        # #preprocess
        # image_array = np.array(image)
        # image = Image.fromarray(preprocess_image(image))

        # Calculate sharpness
        sharpness_image = calculate_sharpness(image)

        # Power-law normalization
        
        st.image(invert_colors(sharpness_image), caption='Inverted Sharpness Map', use_column_width=True)

        sharpness_array = np.array(sharpness_image)
        denoised_image_array = reduce_noise(sharpness_array, method='median')

        denoised_image = Image.fromarray(denoised_image_array)
        st.image(invert_colors(denoised_image), caption='Inverted Denoised Sharpness Map', use_column_width=True)

        enhanced_image = enhance_contrast(denoised_image)
        st.image(invert_colors(enhanced_image), caption='Inverted enhanced Sharpness Map', use_column_width=True)

        binary_image_array = np.array(invert_colors(enhanced_image.convert('L')))
        threshold = 140
        binary_image_array = apply_binary_threshold(binary_image_array, threshold)
        binary_image = Image.fromarray(binary_image_array)
        st.image(binary_image, caption='Binary Thresholded Image', use_column_width=True)

        denoised_image_array = reduce_noise(binary_image_array, method = 'median')
        denoised_image = Image.fromarray(denoised_image_array)
        st.image(denoised_image, caption='Inverted Denoised Sharpness Map', use_column_width=True)

        image = np.array(enhanced_image)

        # Check the number of channels
        if image.ndim == 2:
            # Convert grayscale (L mode) to BGR by replicating the single channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3:
            if image.shape[2] == 4:
                # Convert RGBA to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif image.shape[2] == 3:
                # Convert RGB to BGR
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        sobel = sobel_edge_filter(image)
        st.image(sobel, caption='sobel', use_column_width=True)

        binary_image_array = np.array(sobel)
        threshold = 140
        binary_image_array = apply_binary_threshold(binary_image_array, threshold)
        binary_image = Image.fromarray(binary_image_array)
        st.image(binary_image, caption='Binary Thresholded Image', use_column_width=True)


    
if __name__ == "__main__":
    main()
