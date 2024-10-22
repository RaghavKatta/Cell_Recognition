import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
import random
from scipy.ndimage import binary_fill_holes
import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from scipy.interpolate import interp1d
import json
from scipy.signal import convolve2d, wiener
from scipy.ndimage import gaussian_filter
from numpy.fft import fft2, ifft2
from scipy.signal import fftconvolve
from skimage.restoration import unsupervised_wiener
import matplotlib.pyplot as plt 
from skimage import color, data, restoration
from streamlit_drawable_canvas import st_canvas

@st.cache_data
def sobel_edge_filter(image):
    # Convert the image to grayscale
   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate the magnitude of the gradients
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Normalize the magnitude to 0-255
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
    return magnitude.astype(np.uint8)
@st.cache_data
def binarize_image(image_array, threshold=50):
    # Apply binary thresholding
    _, binary_image_array = cv2.threshold(image_array, threshold, 255, cv2.THRESH_BINARY)
    return binary_image_array
@st.cache_data
def erode_image(image_array, kernel_size=(5, 5), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    # eroded_image = cv2.dilate(image_array, kernel, iterations=1)
    eroded_image = binary_fill_holes(image_array // 255).astype(np.uint8) * 255
    # eroded_image = cv2.erode(eroded_image, kernel, iterations=1)
    return eroded_image

@st.cache_data
def remove_small_features(binary_image, min_size=200):
    if len(binary_image.shape) == 3 and binary_image.shape[2] == 3:
        # Convert to grayscale
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create an empty image to hold the filtered components
    filtered_image = np.zeros_like(binary_image)
    
    # Iterate over each component
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            filtered_image[labels == i] = 255
    
    return filtered_image
@st.cache_data
def find_closest_points(contour1, contour2):
    min_dist = float('inf')
    closest_pair = (None, None)
    for point1 in contour1:
        for point2 in contour2:
            dist = distance.euclidean(point1[0], point2[0])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (point1[0], point2[0])
    return closest_pair
@st.cache_data
def normalize_image(image):
    # Clip and normalize the image to the range [0, 255]
    image = np.clip(image, 0, 255)
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(20,20))
    # contrast_enhanced = clahe.apply(image)
    return image

@st.cache_data
def generate_random_color():
    """Generate a random color."""
    return tuple(random.randint(0, 255) for _ in range(3))

@st.cache_data
def detect_and_draw_connected_segments(image):
    # Convert binary image to 8-bit image
    binary_image = image
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Convert binary image to BGR for coloring
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw filled contours in random colors
    for contour in contours:
        color = generate_random_color()
        cv2.drawContours(color_image, [contour], -1, color, thickness=cv2.FILLED)
        
    
    return color_image
@st.cache_data
def rgb_to_grayscale_average(rgb_image):
    """
    Convert an RGB image to a grayscale image by averaging the RGB values.

    Parameters:
    - rgb_image (np.array): RGB image with shape (height, width, 3).

    Returns:
    - gray_image (np.array): Grayscale image with shape (height, width).
    """
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Input image must be an RGB image with 3 channels.")
    
    # Calculate the average of the RGB values
    gray_image = np.mean(rgb_image, axis=-1).astype(np.uint8)
    
    return gray_image
@st.cache_data
def interpolate_contour_points(contour, min_distance):
    """
    Interpolates additional points in the contour to ensure that points are at most `min_distance` apart.

    Parameters:
    - contour: The contour as a numpy array of shape (N, 1, 2).
    - min_distance: Minimum distance between points in the contour.

    Returns:
    - interpolated_contour: A numpy array of shape (M, 1, 2) with interpolated points.
    """
    points = contour.reshape(-1, 2)
    interpolated_points = []

    for i in range(len(points)):
        start_point = points[i]
        end_point = points[(i + 1) % len(points)]  # Loop back to the start to close the contour

        # Compute the distance between the start and end points
        distance = np.linalg.norm(end_point - start_point)
        num_additional_points = int(np.ceil(distance / min_distance)) - 1

        # Add interpolated points
        for t in np.linspace(0, 1, num_additional_points + 2):
            new_point = (1 - t) * start_point + t * end_point
            interpolated_points.append(new_point.astype(int))

    interpolated_contour = np.array(interpolated_points, dtype=np.int32).reshape(-1, 1, 2)
    return interpolated_contour

class ContourFeatureExtractor:
    def __init__(self):
        self.index_counter = 0
    
    def get_contour_features(self, contour, pixel_to_unit_conversion=1):
        """
        Calculate the features of a contour, including area, centroid, and roundness,
        and assign an index to the contour.
        
        :param contour: The contour to calculate the features for.
        :param pixel_to_unit_conversion: The conversion factor from square pixels to the desired unit.
        :return: A dictionary containing the contour's index, area, centroid, and roundness.
        """
        self.index_counter += 1
        index = self.index_counter
        
        area = self.get_contour_area(contour, pixel_to_unit_conversion)
        centroid = self.get_contour_centroid(contour)
        roundness = self.get_contour_roundness(contour)
        perimeter = self.get_contour_perimeter(contour)
        
        x, y, w, h = cv2.boundingRect(contour)
        bounding_box = (x, y, w, h)
        
         # Calculate Feret diameter
        feret_diameter = self.get_feret_diameter(contour)
        
        return {
            "index": index,
            "area": area,
            "centroid": centroid,
            "roundness": roundness,
            "bounding_box": bounding_box,
            "perimeter": perimeter, 
            'contour' : json.dumps(contour.tolist()), 
            'feret_diameter': feret_diameter
        }
    
    def get_feret_diameter(self, contour):
        """
        Calculate the Feret diameter, the maximum distance between any two points along the contour.
        """
        # Get the convex hull of the contour for more accuracy
        hull = cv2.convexHull(contour)
        
        # Calculate the maximum distance between any two points in the convex hull
        max_distance = 0
        for i in range(len(hull)):
            for j in range(i + 1, len(hull)):
                point1 = hull[i][0]
                point2 = hull[j][0]
                distance = np.linalg.norm(point1 - point2)
                if distance > max_distance:
                    max_distance = distance
        
        return max_distance
    
    def get_contour_perimeter(self, contour):
        """
        Calculate the perimeter of the contour.
        """
        return cv2.arcLength(contour, True) * 1.625
    
    def get_contour_area(self, contour, pixel_to_unit_conversion = 1):
        """
        Calculate the area of a contour and convert it to a different unit.
        
        :param contour: The contour to calculate the area for.
        :param pixel_to_unit_conversion: The conversion factor from square pixels to the desired unit.
        :return: The area of the contour in the desired unit.
        """
        area_in_pixels = cv2.contourArea(contour)
        area_in_units = area_in_pixels * pixel_to_unit_conversion
        return area_in_units
    
    def get_contour_centroid(self, contour):
        """
        Calculate the centroid of a contour.
        
        :param contour: The contour to calculate the centroid for.
        :return: A tuple (cX, cY) representing the centroid coordinates.
        """
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0
        return cX, cY
   
    def get_contour_roundness(self, contour):
        """
        Calculate the roundness of a contour.
        
        :param contour: The contour to calculate the roundness for.
        :return: The roundness value of the contour.
        """
        area = self.get_contour_area(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter != 0:
            roundness = (4 * np.pi * area) / (perimeter ** 2)
        else:
            roundness = 0
        return roundness


def find_best_split_location(contour, distance_threshold =10, min_gap=20):
    """
    Find the best split location in a contour based on distance and gap criteria.

    Parameters:
    contour: The contour as a numpy array of shape (N, 1, 2).
    distance_threshold: The maximum distance between non-adjacent points to consider a potential split.
    min_gap: The minimum distance between the split point and its adjacent points to qualify as a split.

    Returns:
    best_split: A tuple with the indices of the points that are considered the best split location.
    """
    # Extract points from the contour
    points = contour.reshape(-1, 2)
    num_points = len(points)
    best_split = None
    best_dist = float('inf')

    # Calculate distances between non-adjacent points
    for i in range(num_points):
        for j in range(i + min_gap, i + num_points - min_gap):  # Skip adjacent points and ensure cyclicity
            j = j % num_points  # Ensure cyclic nature

            dist = np.linalg.norm(points[i] - points[j])
    if dist < distance_threshold:
        # Calculate vngaps before and after the potential split
        prev_dist_i = np.linalg.norm(points[i] - points[(i - 1) % num_points])
        next_dist_i = np.linalg.norm(points[i] - points[(i + 1) % num_points])
        prev_dist_j = np.linalg.norm(points[j] - points[(j - 1) % num_points])
        next_dist_j = np.linalg.norm(points[j] - points[(j + 1) % num_points])

        # Check if the split location is valid
        if min(prev_dist_i, next_dist_i) > min_gap and min(prev_dist_j, next_dist_j) > min_gap:
            if dist < best_dist:
                best_dist = dist
                best_split = (i, j)

    return best_split
@st.cache_data
def calculate_group_length(group, contour):
    """f
    Calculate the total length of a group of line segments.

    Parameters:
        group (List[int]): The group of indices.
        contour (np.ndarray): The contour to analyze (Nx1x2 shape).

    Returns:
        float: The total length of the group.
    """
    total_length = 0.0
    for i in range(1, len(group)):
        total_length += np.linalg.norm(contour[group[i]] - contour[group[i - 1]])
    return total_length
@st.cache_data
def group_line_segments(contour, distance_threshold=10):
    """
    Groups line segments in a contour based on proximity.

    Parameters:
        contour (np.ndarray): The contour to analyze (Nx1x2 shape).
        distance_threshold (float): The maximum distance between segments to consider them as part of the same group.

    Returns:
        List[List[int]]: A list of grouped line segment indices.
    """
    contour = contour.squeeze()
    groups = []
    current_group = [0]

    for i in range(1, len(contour)):
        if np.linalg.norm(contour[i] - contour[i - 1]) <= distance_threshold:
            current_group.append(i)
        else:
            groups.append(current_group)
            current_group = [i]
    
    if current_group:
        groups.append(current_group)
    
    return groups
@st.cache_data
def evaluate_impact_of_removal(contour, groups, largest_group_index):
    """
    Evaluates the impact of removing each group of line segments from the contour.

    Parameters:
        contour (np.ndarray): The contour to analyze (Nx1x2 shape).
        groups (List[List[int]]): The grouped line segment indices.
        largest_group_index (int): The index of the largest group.

    Returns:
        int: The index of the group that has the most impact when removed.
    """
    max_deviation = 0
    most_deviating_group_index = -1
    original_area = cv2.contourArea(contour)

    for i, group in enumerate(groups):
        if i == largest_group_index:
            continue  # Skip the largest group

        reduced_contour = np.delete(contour, group, axis=0)
        reduced_area = cv2.contourArea(reduced_contour)

        deviation = abs(original_area - reduced_area)
        
        if deviation > max_deviation:
            max_deviation = deviation
            most_deviating_group_index = i

    return most_deviating_group_index
@st.cache_data
def remove_most_deviating_group(contour, distance_threshold=10):
    """
    Removes the group of line segments that deviates the most from the contour, except for the largest group.

    Parameters:
        contour (np.ndarray): The contour to analyze (Nx1x2 shape).
        distance_threshold (float): The maximum distance between segments to consider them as part of the same group.

    Returns:
        np.ndarray: The contour with the most deviating group removed, except for the largest group.
    """
    groups = group_line_segments(contour, distance_threshold)

    # Calculate the length of each group
    group_lengths = [calculate_group_length(group, contour) for group in groups]
    
    # Identify the largest group by length
    largest_group_index = np.argmax(group_lengths)
    
    # Find the most deviating group, skipping the largest one
    most_deviating_group_index = evaluate_impact_of_removal(contour, groups, largest_group_index)

    if most_deviating_group_index != -1:
        group_to_remove = groups[most_deviating_group_index]
        contour = np.delete(contour, group_to_remove, axis=0)
    
    else: 
        return []
    return contour
@st.cache_data
def draw_contour_indices(image, features_list):
    """
    Draw the index number at the centroid of each contour on the image, 
    aligning the center of the text with the centroid and ensuring it stays within the image bounds.
    
    :param image: The image on which to draw.
    :param features_list: A list of dictionaries containing contour features, including index and centroid.
    """
    image_height, image_width = image.shape[:2]

    for features in features_list:
        index = features["index"]
        centroid = features["centroid"]
        cX, cY = centroid

        # Calculate the width and height of the text
        text = str(index)
        font_scale = 2
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 5
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_width, text_height = text_size

        # Calculate the bottom-left corner of the text so that the center aligns with the centroid
        text_x = cX - (text_width // 2)
        text_y = cY + (text_height // 2)

        # Adjust text position if it goes out of bounds
        if text_x < 0:  # Too far left
            text_x = 0
        elif text_x + text_width > image_width:  # Too far right
            text_x = image_width - text_width

        if text_y - text_height < 0:  # Too far up
            text_y = text_height
        elif text_y > image_height:  # Too far down
            text_y = image_height

        # Draw the text on the image
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
@st.cache_data
def detect_and_draw_connected_segments_outline(image, clean, color_image_og):
    # Convert binary image to 8-bit image
    extractor = ContourFeatureExtractor()
    pixel_to_unit_conversion = 1.625 ** 2
    features = []
    binary_image = image
    _, binary_image = cv2.threshold(binary_image, 1, 255, cv2.THRESH_BINARY)
    
    binary_image = cv2.erode(binary_image, (3,3), iterations=1)
    # color_image = binary_image
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Convert binary image to BGR for coloring
    color_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # clean2 = clean
    # Ensure clean2 is grayscale
    clean2 = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) if len(clean.shape) == 3 else clean.copy()
    
    # Define a square kernel (structuring element)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 5x5 square kernel


    clean2 = cv2.dilate(clean2, kernel, iterations=1)
    # clean2 = cv2.dilate(clean2, (5,5), iterations=5)
    clean2 = np.array(clean2)

    # Define the border width
    border_width = 10

    # Get the dimensions of the image
    height, width = clean2.shape[:2]

    # Create a mask where the background is not black
    # Assuming black is 0, you can adjust this if necessary
    black_mask = (clean2 != 0)

    # Create a mask for the border area
    border_mask = np.zeros_like(black_mask, dtype=bool)

    # Top and bottom border
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True

    # Left and right border
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True

    # Combine the border mask with the non-black mask
    final_mask = border_mask & ~black_mask

    # Create a white border color
    border_color = [100]

    # Apply the white border only where the final mask is True
    clean2[final_mask] = border_color

    color_image_gray = cv2.cvtColor(color_image_og, cv2.COLOR_RGB2GRAY)
    # Apply a binary threshold
    _, thresholded_image = cv2.threshold(color_image_gray, 20, 255, cv2.THRESH_BINARY)

    # Fill holes in the thresholded binary image using binary_fill_holes from scipy
    filled_image = binary_fill_holes(thresholded_image).astype(np.uint8) * 255
    filled_image = cv2.dilate(filled_image, kernel, iterations=5)
    filled_array = np.array(filled_image)

    filled_image = cv2.cvtColor(filled_image, cv2.COLOR_GRAY2BGR)
    

    color_image = cv2.cvtColor(color_image_og, cv2.COLOR_RGB2GRAY)
    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)

    # Draw filled contours in random colors
    for contour in contours:
        # Filter out points where x > 50
        # print(contour[0][0])
        # print(len(clean2))
        # print(clean2.shape)
        # print(image.shape)
        filtered_contour = np.array([point for point in contour if clean2[point[0][1], point[0][0]] > 150 ], dtype=np.int32)
        # print(filtered_contour)
        
        color = (255, 255, 255)
        if filtered_contour.size/contour.size >= .8:
            # cv2.drawContours(color_image, [filtered_contour], -1, color, thickness=15)
            cv2.drawContours(color_image, [contour], -1, color, thickness=cv2.FILLED)
            features.append(extractor.get_contour_features(contour, pixel_to_unit_conversion))
        elif filtered_contour.size/contour.size <= .05:
            # cv2.drawContours(color_image, [filtered_contour], -1, color, thickness=15)
            cv2.drawContours(color_image, [contour], -1, (0,0,0), thickness=cv2.FILLED)
            cv2.drawContours(color_image, [contour], -1, (0,0,0), thickness=15)
        elif filtered_contour.size/contour.size >=.05:
            # cv2.drawContours(color_image, [contour], -1, color, thickness = 10)
            # cv2.drawContours(color_image, [filtered_contour], -1, color, thickness = 3)
            realish_area = cv2.contourArea(contour)
            predictedish_area = cv2.contourArea(filtered_contour)
            try:
                if predictedish_area/realish_area > .8:
                    cv2.drawContours(color_image, [contour], -1, color, thickness=cv2.FILLED)
                    features.append(extractor.get_contour_features(contour, pixel_to_unit_conversion))
                else:
                    print('got here')
                    interpolated_filtered_contour = interpolate_contour_points(filtered_contour, 1)
                    penalty = 0
                    for point in interpolated_filtered_contour: 
                        if filled_array[point[0][1], point[0][0]] == 0:
                            penalty += 10
                    # print("penalties: " + str(penalty))
                    double_filtered_contour = np.array([point for point in interpolated_filtered_contour if clean2[point[0][1], point[0][0]] > 0], dtype=np.int32)
                    # print((double_filtered_contour.size - penalty)/interpolated_filtered_contour.size)
                    # if (double_filtered_contour.size - penalty)/interpolated_filtered_contour.size >= .85:
                    cv2.drawContours(color_image, [interpolated_filtered_contour], -1, color, thickness=cv2.FILLED)
                    features.append(extractor.get_contour_features(interpolated_filtered_contour, pixel_to_unit_conversion))
                    # pass
                    # else:
                    #     cv2.drawContours(color_image, [interpolated_filtered_contour], -1, color, thickness= 5)
                    #     pass
                        # cleaned_contour = cleaned_contour = filtered_contour
                        # index = 0
                        # while len(cleaned_contour) > 1 and cv2.contourArea(cleaned_contour)/cv2.contourArea(contour) > .1:
                        #     print("out")
                        #     print(len(cleaned_contour))
                        #     try:
                        #         index += 1
                        #         print(len(cleaned_contour))
                        #         print('split')
                        #         cleaned_contour = remove_most_deviating_group(cleaned_contour, 10)
                        #         print(len(cleaned_contour))
                        #         # cv2.drawContours(color_image, [cleaned_contour], -1, color, thickness = 3)
                        #         interpolated_filtered_contour = interpolate_contour_points(cleaned_contour, 1)
                        #         penalty = 0
                        #         for point in interpolated_filtered_contour: 
                        #             if filled_array[point[0][1], point[0][0]] == 0:
                        #                 penalty += 100
                        #         # print("penalties: " + str(penalty))
                        #         double_filtered_contour = np.array([point for point in interpolated_filtered_contour if clean2[point[0][1], point[0][0]] > 0], dtype=np.int32)
                        #         #print((double_filtered_contour.size - penalty)/interpolated_filtered_contour.size)
                        #         if (double_filtered_contour.size - penalty)/interpolated_filtered_contour.size >= .8 and cv2.contourArea(double_filtered_contour)/cv2.contourArea(contour) > .1:
                        #             cv2.drawContours(color_image, [interpolated_filtered_contour], -1, color, thickness=cv2.FILLED)
                        #             features.append(extractor.get_contour_features(interpolated_filtered_contour, pixel_to_unit_conversion))
                        #             break

                        #     except:
                        #         index += 1
                        #         break
                    # cv2.drawContours(color_image, [interpolated_filtered_contour], -1, color, thickness=cv2.FILLED)
            except:
                pass
        # cv2.drawContours(color_image, [contour], -1, color, thickness=2)
    # draw_contour_indices(color_image, features)
    print(clean2[50, 50])
    print("new______")
    return color_image, features
@st.cache_data
def draw_line_between_points(image, point1, point2, color=(0, 255, 0), thickness=2):
    cv2.line(image, tuple(point1), tuple(point2), color, thickness)

@st.cache_data
def is_connected_through_gray(start, end, gray_image, threshold=200):
    connectivity_image = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.floodFill(connectivity_image, None, start, 255)
    return cv2.pointPolygonTest(connectivity_image, end, False) >= 0
@st.cache_data
def find_shortest_path_through_gray(start, end, gray_image, white_image):
    h, w = gray_image.shape
    path_image = np.zeros_like(gray_image, dtype=np.uint8)
    cv2.line(path_image, start, end, 255, thickness=1)
    gray_image[white_image == 255] = 0
    path = cv2.distanceTransform(gray_image, cv2.DIST_L2, 5)
    return path
@st.cache_data
def process_image_with_shortest_paths(image_array):
    gray = image_array
    _, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    _, binary_gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    binary_gray = cv2.bitwise_and(binary_gray, cv2.bitwise_not(binary_white))
    contours_white, _ = cv2.findContours(binary_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours_white)):
        for j in range(i + 1, len(contours_white)):
            contour1 = contours_white[i]
            contour2 = contours_white[j]
            closest_pair = find_closest_points(contour1, contour2)
            if closest_pair[0] is not None and closest_pair[1] is not None:
                start_point = tuple(map(int, closest_pair[0]))
                end_point = tuple(map(int, closest_pair[1]))
                shortest_path = find_shortest_path_through_gray(start_point, end_point, binary_gray.copy(), binary_white)
                if shortest_path is not None:
                    draw_line_between_points(image_array, start_point, end_point)
    return image_array

@st.cache_data
def remove_small_segments(image_array, min_size=100):
    """Remove connected segments smaller than `min_size` and within dark pixels."""
    # Convert binary image to 8-bit image
    _, binary_image = cv2.threshold(image_array, 127, 255, cv2.THRESH_BINARY)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create an image to mark valid segments
    valid_segments = np.zeros_like(image_array, dtype=np.uint8)
    
    # Iterate over each component
    for label in range(1, num_labels):  # 0 is the background
        x, y, w, h, area = stats[label]
        if area >= min_size:
            segment = (labels == label)
            if not np.any((image_array >= 100) & segment):  # Check if segment touches non-dark pixels
                continue
            valid_segments[segment] = 255
    
    return valid_segments
@st.cache_data
def process_image(image_array):
    # Convert grayscale image to RGB
    rgb_image = np.stack([image_array] * 3, axis=-1)  # Stack grayscale to create RGB
    
    # Create mask for dark pixels
    dark_pixels = image_array < 100
    
    # Remove small segments
    cleaned_image = remove_small_segments(image_array)
    
    # Convert cleaned image to RGB
    cleaned_rgb_image = np.stack([cleaned_image] * 3, axis=-1)
    
    # Set dark pixels to the cleaned image
    rgb_image[dark_pixels] = cleaned_rgb_image[dark_pixels]
    
    return rgb_image
@st.cache_data
# Define a function to calculate distance between centroids
def centroid_distance(centroid1, centroid2):
    return euclidean(centroid1, centroid2)
@st.cache_data
def interpolate_cluster_data(cluster_data, total_days):
    days_with_data = cluster_data['day'].values
    centroids_with_data = np.stack(cluster_data['centroid'].values)
    areas_with_data = cluster_data['area'].values

    # Interpolate centroids
    centroid_interp = interp1d(days_with_data, centroids_with_data, axis=0, kind='linear', fill_value="extrapolate")
    interpolated_centroids = centroid_interp(np.arange(total_days))

    # Interpolate areas
    area_interp = interp1d(days_with_data, areas_with_data, kind='linear', fill_value="extrapolate")
    interpolated_areas = area_interp(np.arange(total_days))

    return interpolated_centroids, interpolated_areas

st.title("Sobel Edge Filtering with Streamlit")

uploaded_files = st.file_uploader("Choose PNG files", type="png", accept_multiple_files=True)

processed_images_for_export = []
binary_images_for_export = []
if uploaded_files is not None:
    image_stack = []
    cross_image_data = []
    cross_image_cleaned_images = []
    final_data_rows = [] 

    for uploaded_file in uploaded_files:
        # Load the image
        image = Image.open(uploaded_file)
        
        image_mode = image.mode
            # Convert image if necessary
        if image_mode == 'I':
            # Convert to NumPy array
            image_array = np.array(image)
            # Normalize the image data
            image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
            # Scale each pixel 2x along both rows and columns
            image_array = np.repeat(np.repeat(image_array, 2, axis=0), 2, axis=1)
            # Convert back to PIL image
            image = Image.fromarray(image_array)
            # Convert to grayscale ('L') for display purposes
            image = image.convert('L')

        
        image = np.array(image)
        try: 
            image_array = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        except: 
            pass
        try: 
            image_array = np.array(image)
        except: 
            pass

        # Check the type of the PIL image object
        print(type(image))

        # Check the type of the NumPy array created from the image
        print(type(image_array))

        # Check the type of individual pixel values in the NumPy array
        print(type(image[0, 0]))

        # # Step 1: Generate a 2D Gaussian kernel (with radius 1 pixel)
        # def gaussian_kernel(size, sigma=1):
        #     """Generates a 2D Gaussian kernel."""
        #     ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        #     xx, yy = np.meshgrid(ax, ax)
        #     kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma)) * 255
        #     return kernel 

        # # Generate Gaussian kernel with size 3x3 (radius 1 pixel, sigma=1)
        # gaussian_kernel_2d = gaussian_kernel(size=image_array.shape[0], sigma=5)
        # # st.image(gaussian_kernel_2d, caption="kernel", use_column_width=True)

        # # # Normalize the deconvolved image to the range of 0-255
        # blurred_image = fftconvolve(image_array, gaussian_kernel_2d, mode='same')

        # blurred_image = (blurred_image - np.min(blurred_image)) / (np.max(blurred_image) - np.min(blurred_image)) * 255
        # blurred_image = blurred_image.astype(np.uint8)
        # st.image(blurred_image, caption = "blurred")
        # Step 2: Perform deconvolution using Wiener filter
        # def wiener_deconvolution(blurred, psf, K=0.01):
        #     """Performs Wiener deconvolution using FFT."""
        #     # Convert both the blurred image and PSF to frequency domain
        #     blurred_fft = fft2(blurred)
        #     psf_fft = fft2(psf, s=blurred.shape)

        #     # Wiener filter in the frequency domain
        #     psf_fft_conj = np.conj(psf_fft)
        #     wiener_filter = psf_fft_conj / (np.abs(psf_fft) ** 2 + K)

        #     # Multiply by the Fourier transform of the blurred image
        #     deconvolved_fft = wiener_filter * blurred_fft

        #     # Convert back to the spatial domain
        #     deconvolved = np.abs(ifft2(deconvolved_fft))
            
        #     return deconvolved
        
        # # Step 4: Apply the Wiener deconvolution with the Gaussian kernel as the PSF
        # # deconvolved_image = wiener_deconvolution(image_array, gaussian_kernel_2d, K=1)
        # deconvolved_image = restoration.wiener(image_array, gaussian_kernel_2d, balance = 1)
        # #deconvolved_image = wiener(image_array, mysize=(3, 3))

        # # Normalize the deconvolved image to the range of 0-255
        # deconvolved_image = (deconvolved_image - np.min(deconvolved_image)) / (np.max(deconvolved_image) - np.min(deconvolved_image)) * 255


        # ###STANDARD NORMALIZATION###
        # #SUBTRACT MEAN/STANDARD DEVIATION - NOT GOOD FOR EDGES?
        # #

        # # Convert the image back to uint8 format (if the original image was uint8)
        # deconvolved_image = deconvolved_image.astype(np.uint8)
        # deconvolved_image_pil = Image.fromarray(deconvolved_image)

        # # Convert the PIL image to grayscale ('L' mode)
        # deconvolved_image = deconvolved_image_pil.convert('L')
        
        # Convert back to the same format as before
        # image = np.array(deconvolved_image)
        # fig, ax = plt.subplots()
        
        # ax.imshow(image)
        # # Display the plot in Streamlit
        # st.title("Matplotlib Plot in Streamlit")
        # st.pyplot(fig)


        # image_array = image_array * 255
        # Display the original image
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # image = np.array(deconvolved_image)
        # # image = image * 255
        # st.image(deconvolved_image, caption='Deconvolved Uploaded Image', use_column_width=True)
        
        # Check the number of channels
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
                
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        # # Apply CLAHE
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # enhanced_image = clahe.apply(gray)
        # st.image(enhanced_image, caption= "clahe")

        enhanced_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        edges = sobel_edge_filter(image)
        st.image(edges, caption= "iniital sobel")
        binary_edges = binarize_image(edges, threshold=35)
        st.image(binary_edges, caption= "iniital binary")
        eroded_image = erode_image(binary_edges)
        cleaned_image = remove_small_features(eroded_image, min_size=300)  # Adjust min_size as needed
        cleaned_copy = cleaned_image.copy()

        edges_pil = Image.fromarray(edges)
        binary_edges_pil = Image.fromarray(binary_edges)
        eroded_pil = Image.fromarray(eroded_image)
        cleaned_image_pil = Image.fromarray(cleaned_image)

        # st.image(edges_pil, caption='Sobel Edge Filtered Image', use_column_width=True)
        # st.image(binary_edges_pil, caption='Binarized Sobel Edge Filtered Image', use_column_width=True)
        # st.image(eroded_pil, caption='eroded & dilated Image', use_column_width=True)
        # st.image(cleaned_image_pil, caption='Cleaned Image with Small Feature Removal', use_column_width=True)

        dark_pixels = image_array < 90
        white_pixels = cleaned_image > 250
        normalized_array =  normalize_image(image_array)
        print("Shape of image_array:", image_array.shape)
        print("Shape of cleaned_image:", cleaned_image.shape)
        print("Shape of normalized_array:", normalized_array.shape)
        print("Shape of dark_pixels:", dark_pixels.shape)
        print("Shape of white_pixels:", white_pixels.shape)
        cleaned_image[dark_pixels] = (255- normalized_array[dark_pixels])/2
        cleaned_image[white_pixels] = 255
        cleaned_image_pil = Image.fromarray(cleaned_image)
        # st.image(cleaned_image_pil, caption='Cleaned Image with Small Feature Removal', use_column_width=True)

        # processed_image = process_image_with_shortest_paths(cleaned_image)
        # processed_image_pil = Image.fromarray(processed_image)
        # st.image(processed_image_pil, caption='Processed Image with Shortest Paths Between Closest Points', use_column_width=True)

        # Process image
        cross_image_cleaned_images.append(image)
        result_image = detect_and_draw_connected_segments(cleaned_image)
        cleaned_image = result_image
        

        # Convert result image to PIL for display in Streamlit
        result_image_pil = Image.fromarray(result_image)
        st.write("image")

        
        # st.image(result_image_pil, caption='Processed Image with Connected Segments', use_column_width=True)

        
        dark_pixels = image_array < 90
        white_pixels = cleaned_image > 20
        cleaned_image_copy = cleaned_image.copy()
        normalized_array =  normalize_image(image_array)
        normalized_array = np.stack([normalized_array] * 3, axis=-1)  # Convert to RGB

        cleaned_image[dark_pixels] = (255- normalized_array[dark_pixels])/2
        cleaned_image[white_pixels] = cleaned_image_copy[white_pixels]
        cleaned_image_pil = Image.fromarray(cleaned_image)
        
        # st.image(cleaned_image_pil, caption='Cleaned Image with Small Feature Removal', use_column_width=True)

        removal = remove_small_features(cleaned_image, min_size=300)  # Adjust min_size as needed
        removal_pil = Image.fromarray(removal)
        
        # st.image(removal_pil, caption='Cleaned Image with Small Feature Removal', use_column_width=True)

        removal_bgr = np.stack([removal] * 3, axis=-1)
        removed_clean = np.minimum(removal_bgr, cleaned_image)
        removed_clean_pil = Image.fromarray(removed_clean)

        # st.image(removed_clean_pil, caption='Cleaned Image with Small Feature Removal', use_column_width=True)

        removal, features = detect_and_draw_connected_segments_outline(removal, cleaned_copy, removed_clean)
        removal_pil = Image.fromarray(removal)

        st.image(removal_pil, caption='Cleaned Image with Small Feature Removal', use_column_width=True)
        st.dataframe(features)
        features = pd.DataFrame(features)
        cross_image_data.append(features)

    # Convert list of DataFrames to a single DataFrame
    cross_image_data = pd.concat(cross_image_data, keys=range(len(cross_image_data)), names=['day', 'cell'])

    # Initialize list to store centroids
    all_centroids = []

    # Iterate through the cross_image_data
    for (day, cell), row in cross_image_data.iterrows():
        if row['area'] >= 300:  # Check if the area is 300 or more
            all_centroids.append({
                'day': day,
                'index': cell,
                'centroid': row['centroid'],
                'area': row['area'],
                'roundness': row['roundness'],
                'bounding box': row['bounding_box'], 
                'perimeter': row['perimeter'], 
                'contour' : row['contour'], 
                'feret_diameter': row['feret_diameter'], 
            })

    #Convert the list to a dataframe for easier processing
    centroids_df = pd.DataFrame(all_centroids)
    threshold_distance = 30
    min_days_for_cluster = 3
        
        
    # Apply clustering based on centroid proximity
    clustering = DBSCAN(eps=threshold_distance, min_samples=min_days_for_cluster).fit(np.stack(centroids_df['centroid'].values))
    centroids_df['cluster_id'] = clustering.labels_

    # Filter out noise (unclustered points)
    valid_clusters = centroids_df[centroids_df['cluster_id'] != -1]

    # Group by cluster_id and check the number of unique days in each cluster
    cluster_summary = valid_clusters.groupby('cluster_id')['day'].nunique().reset_index()
    cluster_summary = cluster_summary[cluster_summary['day'] >= min_days_for_cluster]

    # Filter valid clusters based on the criteria
    valid_clusters = valid_clusters[valid_clusters['cluster_id'].isin(cluster_summary['cluster_id'])]
    #Step 1: Initial Clustering with Small `eps`
    small_eps = 40  # Small initial threshold distance
    min_days_for_initial_cluster = 3  # Initial minimum number of days for clusters
    min_days_for_final_cluster = 5  # Final minimum number of days for clusters
    max_eps = 100  # Maximum eps value to limit growth
    eps_increment = 20  # Increment to increase eps in each iteration

    centroids_df = pd.DataFrame(all_centroids)
    clustering = DBSCAN(eps=small_eps, min_samples=min_days_for_initial_cluster).fit(np.stack(centroids_df['centroid'].values))
    centroids_df['cluster_id'] = clustering.labels_

    # Initial valid clusters (without filtering for days)
    valid_clusters = centroids_df[centroids_df['cluster_id'] != -1]

    # Step 2: Iteratively Expand Search Radius and Update Cluster Assignments
    current_eps = small_eps

    # Track which centroids have already been assigned to clusters
    assigned_centroids = set(valid_clusters.index)

    while current_eps <= max_eps:
        for cluster_id in valid_clusters['cluster_id'].unique():
            cluster_data = valid_clusters[valid_clusters['cluster_id'] == cluster_id]
            missing_days = set(range(len(uploaded_files))) - set(cluster_data['day'].unique())
            
            if missing_days:
                # Expand search radius for missing days
                for missing_day in missing_days:
                    missing_day_data = centroids_df[centroids_df['day'] == missing_day]
                    
                    # Filter out already assigned centroids from consideration
                    available_indices = missing_day_data.index.difference(assigned_centroids)
                    available_data = missing_day_data.loc[available_indices]

                    if available_data.empty:
                        continue  # Skip if no available centroids

                    # Compute distances from existing cluster centroids to points in the missing day
                    for centroid in cluster_data['centroid']:
                        distances = np.linalg.norm(np.stack(available_data['centroid'].values) - centroid, axis=1)
                        
                        # Find the closest point within the expanded_eps radius
                        if np.any(distances <= current_eps):
                            closest_point_idx = np.argmin(distances)
                            closest_point_global_idx = available_data.index[closest_point_idx]
                            
                            # Assign only the closest point to the cluster if it's within the radius
                            if distances[closest_point_idx] <= current_eps:
                                centroids_df.loc[closest_point_global_idx, 'cluster_id'] = cluster_id
                                assigned_centroids.add(closest_point_global_idx)
        # # Check for new clusters with the expanded eps value
        # new_clustering = DBSCAN(eps=current_eps, min_samples=4).fit(np.stack(centroids_df[centroids_df['cluster_id'] == -1]['centroid'].values))
        
        # # Assign new cluster IDs only to points that were noise in previous iterations
        # new_cluster_ids = new_clustering.labels_
        # noise_indices = centroids_df[centroids_df['cluster_id'] == -1].index
        # centroids_df.loc[noise_indices, 'cluster_id'] = new_cluster_ids
        
        # # Re-check for valid clusters including any new ones
        # valid_clusters = centroids_df[centroids_df['cluster_id'] != -1]
        # Increase eps for the next iteration
        current_eps += eps_increment

    # Step 3: Final Filtering of Valid Clusters Based on the Final Criteria
    # Re-filter valid clusters after all expansions
    valid_clusters = centroids_df[centroids_df['cluster_id'] != -1]
    
   # Group by cluster_id and check the number of unique days in each cluster
    cluster_summary = valid_clusters.groupby('cluster_id')['day'].nunique().reset_index()
    valid_cluster_ids = cluster_summary[cluster_summary['day'] >= min_days_for_final_cluster]['cluster_id']

    # Reassign cluster IDs that do not meet the final criteria to -1
    centroids_df.loc[~centroids_df['cluster_id'].isin(valid_cluster_ids), 'cluster_id'] = -1

    # Filter valid clusters based on the final criteria
    valid_clusters = centroids_df[centroids_df['cluster_id'] != -1]

    # Identify points that were not included in the final valid clusters (remaining noise)
    remaining_noise = centroids_df[centroids_df['cluster_id'] == -1]
    # Apply DBSCAN on these points with eps=100 and min_samples=4
    if not remaining_noise.empty:
        # Perform DBSCAN on the remaining noise points
        new_clustering = DBSCAN(eps=80, min_samples=4).fit(np.stack(remaining_noise['centroid'].values))
        
        # Assign new cluster IDs only to points that were noise in previous iterations
        new_cluster_ids = new_clustering.labels_
        noise_indices = remaining_noise.index

        # Find the maximum existing cluster ID
        max_existing_cluster_id = centroids_df['cluster_id'].max()

        # Offset the new cluster IDs to ensure they are unique
        new_cluster_ids = np.where(new_cluster_ids != -1, new_cluster_ids + max_existing_cluster_id + 1, -1)
        
        # Update centroids_df with the new cluster IDs (ignore noise labeled as -1)
        centroids_df.loc[noise_indices, 'cluster_id'] = new_cluster_ids

    # Update the valid_clusters DataFrame with the newly found clusters
    valid_clusters = centroids_df[centroids_df['cluster_id'] != -1]
    # Initialize final dataframe
    final_df = pd.DataFrame()

    # For each valid cluster, compile the information from all days
    for cluster_id in valid_clusters['cluster_id'].unique():
        cluster_data = valid_clusters[valid_clusters['cluster_id'] == cluster_id]
        cluster_row = {'cluster_id': cluster_id}
        
        for day in range(len(uploaded_files)):
            day_data = cluster_data[cluster_data['day'] == day]
            if not day_data.empty:
                day_data = day_data.iloc[0]  # Take the first entry (if multiple, they are close and represent the same cell)
                cluster_row.update({
                    f'centroid_day{day}': day_data.get('centroid', np.nan),
                    f'area_day{day}': day_data.get('area', np.nan),
                    f'roundness_day{day}': day_data.get('roundness', np.nan),
                    # f'bounding_box{day}': day_data.get('bounding_box', np.nan),  # Use .get() for safety
                    f'perimeter{day}': day_data.get('perimeter', np.nan),
                    f'contour{day}': day_data.get('contour', np.nan),
                    f'feret_diameter{day}': day_data.get('feret_diameter', np.nan)
                })
            else:
                cluster_row.update({
                    f'centroid_day{day}': np.nan,
                    f'area_day{day}': np.nan,
                    f'roundness_day{day}': np.nan,
                    # f'bounding_box{day}': np.nan, 
                    f'perimeter{day}': np.nan, 
                    f'contour{day}': np.nan, 
                    f'feret_diameter{day}': np.nan, 
                })
        
        # Append the row dictionary to the final_data_rows list
        final_data_rows.append(cluster_row)

    # Convert the list of rows into a final DataFrame
    final_df = pd.DataFrame(final_data_rows)

    # Display the final dataframe
    st.dataframe(final_df)


    # Iterate over each image and corresponding data
    for day, image in enumerate(cross_image_cleaned_images):
        # Create a copy of the image to draw the circles on
        if len(image.shape) == 2:  # Check if the image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        overlay = image.copy()
        
        # Create a blank white image for contours (same size as the original)
        blank_image = np.ones_like(image) * 255  # White background

        # Iterate over each row in the final dataframe for the corresponding day
        for _, row in final_df.iterrows():
            centroid = row[f'centroid_day{day}']
            area = row[f'area_day{day}']
            contour_str = row.get(f'contour{day}', None) 

            
            if not pd.isna(centroid) and not pd.isna(area):
                # Convert centroid to integer tuple
                centroid = tuple(map(int, centroid))
                # Calculate radius based on area (assuming circular area)
                # ratio = 1.625 ** 2
                # area = area/ratio
                radius = int(np.sqrt(area / np.pi))

                # Draw a semi-transparent circle
                color = (0, 255, 0)  # Green color
                transparency = 0.1  # Transparency factor
                
                # Draw filled circle on overlay
                cv2.circle(overlay, centroid, radius, color, thickness=-1)
            # If a contour is available, parse and draw it
                if contour_str and not pd.isna(contour_str):
                    try:
                        # Convert the contour string back to a NumPy array
                        contour = np.array(json.loads(contour_str), dtype=np.int32)
                        # print(f'contour string: {contour}')

                        # Ensure the contour is valid (a list of points)
                        if isinstance(contour, (list, np.ndarray)):
                            contour = np.array(contour, dtype=np.int32)
                            # print("was deemed valid")
                            # Draw the contour and fill it in black
                            cv2.drawContours(image, [contour], -1, (255, 0, 0), thickness=2)

                            # Draw the contour on the blank white image (black contour on white background)
                            cv2.drawContours(blank_image, [contour], -1, (0, 0, 0), thickness=2)

                    except Exception as e:
                        print(f"Error parsing contour: {e}")

        # Blend the overlay with the original image
        cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)
        
        # Convert the image back to PIL format for display in Streamlit
        result_image_pil = Image.fromarray(image)
        st.image(result_image_pil, caption=f'Processed Image with Cell Recognition Overlay - Day {day}', use_column_width=True)
      
        contour_image_pil = Image.fromarray(blank_image)
        binary_images_for_export.append(contour_image_pil)
        # Display the image with black contours on a white background
        st.image(contour_image_pil, caption=f'Binary Image (Black on White) - Day {day}', use_column_width=True)
    # Assuming final_df is your DataFrame containing the cluster data
    # anomalies = []

    # # Iterate through the clusters in final_df
    # for _, row in final_df.iterrows():
    #     for day in range(len(uploaded_files)):
    #         centroid_key = f'centroid_day{day}'
    #         area_key = f'area_day{day}'
            
    #         # Check if both the centroid and area for the current day are not NaN
    #         if pd.notna(row[centroid_key]) and pd.notna(row[area_key]):
    #             current_area = row[area_key]
                
    #             # Collect areas and corresponding days where data is present
    #             other_days_areas = []
    #             other_days = []
    #             for i in range(len(uploaded_files)):
    #                 if i != day and pd.notna(row[f'area_day{i}']):
    #                     other_days.append(i)
    #                     other_days_areas.append(row[f'area_day{i}'])

    #             if other_days_areas:  # Ensure there are valid areas to compare with
    #                 # Perform interpolation
    #                 other_days = np.array(other_days)
    #                 other_days_areas = np.array(other_days_areas)
                    
    #                 # Interpolate to predict the area for the current day
    #                 interpolated_area = np.interp(day, other_days, other_days_areas)
                    
    #                 lower_bound = 0.25 * interpolated_area
    #                 upper_bound = 2 * interpolated_area
                    
    #                 # Flag as anomaly if the area is 75% below or 100% above the predicted area
    #                 if current_area < lower_bound or current_area > upper_bound:
    #                     anomalies.append({
    #                         'cluster_id': row['cluster_id'],
    #                         'day': day,
    #                         'centroid': row[centroid_key],
    #                         'area': current_area,
    #                         'predicted_area': interpolated_area,
    #                         'roundness': row[f'roundness_day{day}'],
    #                     })


    # # Create a DataFrame for anomalies
    # anomalies_df = pd.DataFrame(anomalies)
    # st.title("anomalies")

    # Existing Data Processing Code (Assuming this is to display the valid clusters)
    # st.dataframe(anomalies_df)


# This merging_clusters_df will contain the clusters suspected of merging.
    @st.cache_data
    def interpolate_cluster_data(final_df, total_days):
        # Iterate over each row (cluster) in the final_df
        for index, row in final_df.iterrows():
            # Prepare lists to hold the existing data and corresponding days
            days_with_data = []
            centroids = []
            areas = []
            roundness_values = []

            # Collect existing data
            for day in range(total_days):
                centroid_key = f'centroid_day{day}'
                area_key = f'area_day{day}'
                roundness_key = f'roundness_day{day}'

                if pd.notna(row[centroid_key]) and pd.notna(row[area_key]) and pd.notna(row[roundness_key]):
                    days_with_data.append(day)
                    centroids.append(row[centroid_key])
                    areas.append(row[area_key])
                    roundness_values.append(row[roundness_key])

            # Interpolate for missing days
            for day in range(total_days):
                if day not in days_with_data:
                    # Interpolate centroid
                    interpolated_centroid = np.interp(day, days_with_data, [c[0] for c in centroids]), np.interp(day, days_with_data, [c[1] for c in centroids])
                    final_df.at[index, f'centroid_day{day}'] = interpolated_centroid

                    # Interpolate area
                    interpolated_area = np.interp(day, days_with_data, areas)
                    final_df.at[index, f'area_day{day}'] = interpolated_area

                    # Interpolate roundness
                    interpolated_roundness = np.interp(day, days_with_data, roundness_values)
                    final_df.at[index, f'roundness_day{day}'] = interpolated_roundness

        return final_df

    # Assume total_days is the number of days/images
    total_days = len(uploaded_files)

    # Perform interpolation on final_df
    pre_interpolated_df = final_df
    final_df = interpolate_cluster_data(final_df, total_days)
    @st.cache_data
    def add_centroid_coordinates_columns(df, total_days):
        """
        Add separate columns for centroid X and Y coordinates to the given DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to modify.
        total_days (int): The total number of days (images).

        Returns:
        pd.DataFrame: The modified DataFrame with centroid_x and centroid_y columns added.
        """
        for day in range(total_days):
            centroid_col = f'centroid_day{day}'
            centroid_x_col = f'centroid_x_day{day}'
            centroid_y_col = f'centroid_y_day{day}'

            # Initialize new columns
            df[centroid_x_col] = np.nan
            df[centroid_y_col] = np.nan

            # Split the centroid into x and y if it is not NaN
            for index, row in df.iterrows():
                if pd.notna(row[centroid_col]):
                    # Extract the (x, y) tuple
                    x, y = row[centroid_col]
                    df.at[index, centroid_x_col] = x
                    df.at[index, centroid_y_col] = y
            cols = df.columns.tolist()
            centroid_index = cols.index(centroid_col)

            # Move the new columns right after the centroid column
            cols.insert(centroid_index + 1, cols.pop(cols.index(centroid_x_col)))
            cols.insert(centroid_index + 2, cols.pop(cols.index(centroid_y_col)))

            df = df[cols]  # Reorder the DataFrame columns
        return df

    # Apply the function to both pre_interpolated_df and final_df
    pre_interpolated_df = add_centroid_coordinates_columns(pre_interpolated_df, total_days)
    final_df = add_centroid_coordinates_columns(final_df, total_days)

    # Display the updated DataFrames
    st.dataframe(pre_interpolated_df)
    st.dataframe(final_df)

    # Display the interpolated final_df
    st.title("hello")
    st.dataframe(final_df)
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    # Convert the DataFrame to CSV
    csv = convert_df_to_csv(final_df)

    # Create a download button
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name='final_df.csv',
        mime='text/csv',
    )

    
    @st.cache_data
    def detect_overlapping_and_merging_clusters(final_df, overlap_threshold=0.5, area_change_threshold=1.5):
        merging_clusters = []

        # Iterate over days and clusters
        for day in range(len(uploaded_files)):  # Loop through days except the last one
            clusters_day1 = final_df[[f'centroid_day{day}', f'area_day{day}', 'cluster_id']].dropna()
            clusters_day2 = final_df[[f'centroid_day{day}', f'centroid_day{day}', f'area_day{day}', 'cluster_id']].dropna()

            # Compare clusters between consecutive days
            for _, cluster1 in clusters_day1.iterrows():
                for _, cluster2 in clusters_day2.iterrows():
                    
                    # Skip comparison if it's the same cluster on consecutive days
                    if cluster1['cluster_id'] == cluster2['cluster_id']:
                        continue

                    # Calculate bounding box overlap
                    overlap = calculate_overlap(cluster1[f'area_day{day}'], cluster2[f'area_day{day}'], cluster1[f'centroid_day{day}'], cluster2[f'centroid_day{day}'])
                    
                    # # Calculate area change
                    # area_change = cluster2[f'area_day{day+1}'] / cluster1[f'area_day{day}']
                    
                    # Check if the overlap is significant and the area has changed drastically
                    if overlap > overlap_threshold:
                    
                        merging_clusters.append({
                            'cluster_id_day1': cluster1['cluster_id'],
                            'cluster_id_day2': cluster2['cluster_id'],
                            'day': day,
                            'overlap': overlap,
                            # 'area_change': area_change,
                        })

        return pd.DataFrame(merging_clusters)
    @st.cache_data
    def calculate_overlap(area1, area2, centroid1, centroid2):
         # Calculate the radii from the areas
        r1 = np.sqrt(area1 / np.pi)
        r2 = np.sqrt(area2 / np.pi)
        
        # Calculate the distance between the centroids
        if isinstance(centroid1[0], tuple):
            centroid1 = centroid1[0]
        if isinstance(centroid2[0], tuple):
            centroid2 = centroid2[0]
        d = np.linalg.norm(np.array(centroid1) - np.array(centroid2))
        
        # If the circles do not overlap at all
        if d >= r1 + r2:
            return 0
        
        # If one circle is completely inside the other
        if d <= abs(r1 - r2):
            return 1
        
        # Calculate the overlap area
        part1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
        part2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
        part3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
        
        overlap_area = part1 + part2 - part3
        return overlap_area/(min(area1,area2))
       

    # Detect overlapping and merging clusters
    merging_clusters_df = detect_overlapping_and_merging_clusters(final_df)
    st.title("merging")
    st.dataframe(merging_clusters_df)



    # Iterate over each image and corresponding data
    for day, image in enumerate(cross_image_cleaned_images):
        # Create a copy of the image to draw the circles on
        if len(image.shape) == 2:  # Check if the image is grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        overlay = image.copy()
        
        # Iterate over each row in the final dataframe for the corresponding day
        for _, row in final_df.iterrows():
            centroid = row[f'centroid_day{day}']
            area = row[f'area_day{day}']
            cluster_id = row['cluster_id']  # Assuming 'cluster_id' is a column in final_df

            if not pd.isna(centroid) and not pd.isna(area):
                # Convert centroid to integer tuple
                centroid = tuple(map(int, centroid))
                # Calculate radius based on area (assuming circular area)
                radius = int(np.sqrt(area / np.pi))

                # Draw a semi-transparent circle on the overlay
                color = (255, 0, 0)  # Red color
                transparency = 0.1  # Transparency factor
                
                # Draw filled circle on overlay
                cv2.circle(overlay, centroid, radius, color, thickness=-1)

        # Blend the overlay with the original image to keep the circles semi-transparent
        cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)

        #  # Now, draw circles for the anomalies in blue
        # for _, anomaly in anomalies_df.iterrows():
        #     if anomaly['day'] == day:  # Only draw if the anomaly is on the current day
        #         centroid = anomaly['centroid']
        #         area = anomaly['area']

        #         if not pd.isna(centroid) and not pd.isna(area):
        #             # Convert centroid to integer tuple
        #             centroid = tuple(map(int, centroid))
        #             # Calculate radius based on area (assuming circular area)
        #             radius = 10

        #             # Draw a semi-transparent circle for anomalies
        #             color = (0, 0, 255)  # Blue color
        #             transparency = 0.1  # Transparency factor
                    
        #             # Draw filled circle on overlay for anomalies
        #             cv2.circle(image, centroid, radius, color, thickness=-1)

        # # Blend the overlay with the original image to keep the circles semi-transparent
        # cv2.addWeighted(overlay, transparency, image, 1 - transparency, 0, image)

        # # Draw the text labels (cluster IDs) on the blended image (fully opaque)
        for _, row in final_df.iterrows():
            centroid = row[f'centroid_day{day}']
            cluster_id = row['cluster_id']  # Assuming 'cluster_id' is a column in final_df

            if not pd.isna(centroid):
                # Convert centroid to integer tuple
                centroid = tuple(map(int, centroid))

                # Draw the cluster ID number directly on the final blended image
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = .5
                font_thickness = 2
                text_size, _ = cv2.getTextSize(str(cluster_id), font, font_scale, font_thickness)
                text_x = int(centroid[0] - text_size[0] / 2)
                text_y = int(centroid[1] + text_size[1] / 2)
                
                # Draw the text on the blended image
                cv2.putText(image, str(cluster_id), (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

         # Add dots for merging clusters
        for _, merge_row in merging_clusters_df.iterrows():
            if merge_row['day'] == day:
                cluster_id_day1 = merge_row['cluster_id_day1']
                cluster_id_day2 = merge_row['cluster_id_day2']
                
                # Get the centroid of the cluster for the current day
                centroid_day1 = final_df.loc[final_df['cluster_id'] == cluster_id_day1, f'centroid_day{day}'].values
                centroid_day2 = final_df.loc[final_df['cluster_id'] == cluster_id_day2, f'centroid_day{day}'].values
                
                # Check if the centroids exist and are valid
                if len(centroid_day1) > 0 and not pd.isna(centroid_day1[0]):
                    centroid = tuple(map(int, centroid_day1[0]))
                    cv2.circle(image, centroid, 15, (0, 0, 255), thickness=-1)  # Blue dot for merging cluster
                
                if len(centroid_day2) > 0 and not pd.isna(centroid_day2[0]):
                    centroid = tuple(map(int, centroid_day2[0]))
                    cv2.circle(image, centroid, 15, (0, 0, 255), thickness=-1)  # Blue dot for merging cluster
            
        # Convert the image back to PIL format for display in Streamlit
        result_image_pil = Image.fromarray(image)
        processed_images_for_export.append(result_image_pil)
        st.image(result_image_pil, caption=f'Processed Image with Cell Recognition Overlay - Day {day}', use_column_width=True)

    st.title("need to do:")
    st.write("incorporate more splitting")
    st.write("recheck of images")

    st.write(len(processed_images_for_export))

    import zipfile
    import os
    from io import BytesIO

   
    # Function to save all images as a zip file
    def save_images_as_zip(images, filenames):
        # Create a BytesIO object to store the zip file in memory
        zip_buffer = BytesIO()
        
        # Create a new zip file in the BytesIO buffer
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add each image to the zip file
            for image, filename in zip(images, filenames):
                # Convert the image to RGB if it’s grayscale
                image = np.array(image)
                if len(image.shape) == 2:  # Check if the image is grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                
                # Convert NumPy array to bytes using OpenCV
                _, image_bytes = cv2.imencode('.png', image)
                
                # Write the image to the zip file
                zip_file.writestr(filename, image_bytes.tobytes())
        
        # Return the zip file buffer
        return zip_buffer.getvalue()

    # Assuming 'cross_image_cleaned_images' contains the processed images
    
        # Prepare image filenames
    image_filenames = [f"processed_image_day_{day}.png" for day in range(len(cross_image_cleaned_images))]

    # Get the zip file with all images
    zip_file = save_images_as_zip(processed_images_for_export, image_filenames)
    
    image_filenames = [f"binary_image_day_{day}.png" for day in range(len(cross_image_cleaned_images))]

    binary_zip_file = save_images_as_zip(binary_images_for_export, image_filenames)
    
    # Provide download button for the zip file
    st.download_button(
        label="Download All Processed Images as Zip",
        data=zip_file,
        file_name="processed_images.zip",
        mime="application/zip"
    )

    st.download_button(
        label="Download All Binary Images as Zip",
        data=binary_zip_file,
        file_name="binary_images.zip",
        mime="application/zip"
    )

