import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import io
import zipfile

@st.cache_data
# Function to add a unique index column for each cell, incorporating the Image Day
def add_unique_index(dataframe):
    # Create a unique label by combining Image Day, Cell Identity, and a unique count
    dataframe['Unique Label'] = dataframe.groupby(['Image Day', 'Cell Identity']).cumcount() + 1
    dataframe['Unique Identity'] = (
        dataframe['Image Day'].astype(str) + "_" + 
        dataframe['Cell Identity'].astype(str) + "_" + 
        dataframe['Unique Label'].astype(str)
    )
    
    # Move 'Unique Identity' to the leftmost position
    columns = ['Unique Identity'] + [col for col in dataframe.columns if col != 'Unique Identity']
    dataframe = dataframe[columns]
    
    return dataframe


@st.cache_data
# Function to resize images to 2048x2048
def resize_to_2048(images):
    resized_images = []
    for image_file in images:
        image = Image.open(image_file).convert('RGBA')  # Convert to RGBA for overlay support
        # Resize image to 2048x2048
        resized_image = image.resize((2048, 2048))
        resized_images.append(resized_image)
    return resized_images

@st.cache_data
# Function to apply binary threshold and fill holes with black as walls
def apply_binary_threshold_and_fill_holes(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    dilated_walls = cv2.erode(binary_image, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:  # If it's not an outer contour (i.e., it's a hole)
            cv2.drawContours(binary_image, contours, i, 255, -1)  # Fill the hole with white
    
    filled_image = cv2.bitwise_not(binary_image)
    final_image = Image.fromarray(filled_image)
    return final_image

@st.cache_data
# Function to detect centroids of cells
def find_cell_centroids(binary_image):
    binary_image_cv = np.array(binary_image)
    binary_image_cv = cv2.bitwise_not(binary_image_cv)
    _, binary_image_cv = cv2.threshold(binary_image_cv, 128, 255, cv2.THRESH_BINARY)
    num_labels, labels_im = cv2.connectedComponents(binary_image_cv)
    centroids = []
    contours_list = []

    for label in range(1, num_labels):
        object_mask = np.where(labels_im == label, 255, 0).astype(np.uint8)
        M = cv2.moments(object_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centroids.append((cX, cY))

        # Create the contours from the mask for the current component
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Add the contour of this component to the list
        if contours:
            contours_list.append(contours[0])  # Only take the first contour, as there should be just one
        
    return centroids, contours_list

@st.cache_data
# Function to compute geometric properties of each cell
def compute_geometric_properties(contours, centroids, labels, image_day, pixel_to_mm_ratio=1.625):
    data = []
    for i, (contour, centroid) in enumerate(zip(contours, centroids)):
        area_pixels = cv2.contourArea(contour)
        area_mm = area_pixels 
        perimeter_pixels = cv2.arcLength(contour, True)
        perimeter_mm = perimeter_pixels 
        bounding_rect = cv2.minAreaRect(contour)

        max_feret_diameter = 0
        for pt1 in contour:
            for pt2 in contour:
                dist = np.linalg.norm(pt1 - pt2)
                if dist > max_feret_diameter:
                    max_feret_diameter = dist

        feret_diameter_pixels = max_feret_diameter
        if perimeter_pixels != 0:
            circularity = (4 * np.pi * area_pixels) / (perimeter_pixels ** 2)
        else:
            circularity = 0

        # st.title(labels[i])
        data.append({
            "Cell Identity": float(str(labels[i])),  # Directly using DBSCAN label as Cell Identity
            "Image Day": image_day,
            "Centroid X ": centroid[0],
            "Centroid Y ": centroid[1],
            "Area": area_pixels,
            "Perimeter": perimeter_pixels,
            "Feret Diameter": feret_diameter_pixels,
            "Circularity": circularity
        })
    return pd.DataFrame(data)


# Function to allow CSV download
def download_csv(dataframe, filename):
    csv = dataframe.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )


def download_area_table(dataframe, filename):
    # Reset the index to include 'Cell Identity' as a column
    dataframe = dataframe.reset_index()
    csv = dataframe.to_csv(index=False)
    st.download_button(
        label=f"Download {filename}",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

@st.cache_data
def draw_labels_on_cells(image, centroids, labels, font_size=75):
    if image.mode == 'RGBA':
       
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
        image = image.convert("RGB")
    if image.mode == 'I':
       
        image_array = np.array(image)
        image_array = (image_array / np.max(image_array) * 255).astype(np.uint8)
        image_array = np.repeat(np.repeat(image_array, 2, axis=0), 2, axis=1)
        image = Image.fromarray(image_array)
        image = image.convert('L')
    if image.mode == 'L':
        
        image = image.convert("RGB")
    
    if image.size != (4096, 4096):
        image = image.resize((4096, 4096), Image.Resampling.LANCZOS)


    draw = ImageDraw.Draw(image)
   
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    for i, (x, y) in enumerate(centroids):
        label = str(labels[i])
        draw.text((2*x, 2*y), label, font=font, fill=(255, 0, 0))  # Red label for tracking
    return image

@st.cache_data
def remove_small_features(binary_image, min_size=5):
    # print("got here")
    
    # Convert PIL Image to NumPy array if it's a PIL Image
    if isinstance(binary_image, Image.Image):
        binary_image = np.array(binary_image)
    
    # Convert to grayscale if the image is in color
    if len(binary_image.shape) == 3 and binary_image.shape[2] in [3, 4]:
        binary_image = cv2.cvtColor(binary_image, cv2.COLOR_RGBA2GRAY)

    # Ensure the image is binary (0 and 255 values)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)
    
    # Invert the binary image to treat black segments as connected components
    inverted_image = cv2.bitwise_not(binary_image)

    # Find connected components in the inverted image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_image, connectivity=8)
    
    # Create an empty image to hold the filtered components
    filtered_image = np.zeros_like(binary_image)
    
    # Iterate over each component (skip the background label 0)
    for i in range(1, num_labels):
        # print(f"Component {i} Area: {stats[i, cv2.CC_STAT_AREA]}")
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            # print("got here 2")
            filtered_image[labels == i] = 255
    
    # Invert the filtered image back to its original form (black segments retained)
    filtered_image = cv2.bitwise_not(filtered_image)
    
    return filtered_image


@st.cache_data
# Function to overlay images
def overlay_images(uploaded_images, binary_images):
    num_images = min(len(uploaded_images), len(binary_images))
    final_images = []
    centroids_across_days = []
    contours_across_days = []
    for i in range(num_images):
        print("i" + str(i))
        # Open the images
        img1_pil = uploaded_images[i]
        img2_pil = binary_images[i]

        # Resize img2 to match img1 size
        img2_pil = img2_pil.resize(img1_pil.size)

        # Convert PIL images to NumPy arrays
        img1 = np.array(img1_pil)
        img2 = np.array(img2_pil)

        # Ensure images are in grayscale
        if len(img1.shape) == 4:
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGBA2GRAY)
        else:
            img1_gray = img1.copy()
        if len(img2.shape) == 4:
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGBA2GRAY)
        else:
            img2_gray = img2.copy()

        # Apply binary thresholding
        _, img1_binary = cv2.threshold(img1_gray, 127, 255, cv2.THRESH_BINARY)
        _, img2_binary = cv2.threshold(img2_gray, 127, 255, cv2.THRESH_BINARY)

        # Perform bitwise AND operation
        overlaid_image = cv2.bitwise_and(img1_binary, img2_binary)

        # Convert back to PIL Image for further processing
        overlaid_image_pil = Image.fromarray(overlaid_image)

        # Apply binary thresholding and fill holes
        final_image = apply_binary_threshold_and_fill_holes(overlaid_image_pil)

        # # Remove small features
        final_image = remove_small_features(final_image, min_size=85)

        # Find centroids and contours
        centroids, contours = find_cell_centroids(final_image)

        # Append results to lists
        centroids_across_days.append(centroids)
        contours_across_days.append(contours)
        final_images.append(final_image)

        # Display the image
        st.image(final_image, caption=f"Binarized Image {i+1}")

    return final_images, centroids_across_days, contours_across_days

@st.cache_data
# Function to find and print entries in the pivot table that are zero
def find_zero_entries(pivot_table):
    zero_entries = []
    # Iterate over the pivot table to find cells with zero values
    for cell_identity in pivot_table.index:
        for day in pivot_table.columns:
            if pivot_table.at[cell_identity, day] == 0:
                zero_entries.append((cell_identity, day))
    
    return zero_entries


def update_labeled_images_from_cleaned_data(images, full_data_cleaned):
    st.write("Columns in full_data_cleaned:", full_data_cleaned.columns)
    
    updated_labeled_images = []
    
    # Iterate through each image
    for i, image in enumerate(images):
        # Extract data for the current image day from full_data_cleaned
        current_day = f"Day {i + 1}"
        day_data = full_data_cleaned[full_data_cleaned['Image Day'] == current_day]

        # Extract centroids and labels from day_data
        centroids = list(zip(day_data['Centroid X '], day_data['Centroid Y ']))
        # Combine 'Cell Identity' and 'Unique Label' columns into a new column
        combined_labels = day_data['Cell Identity'].astype(int).astype(str) + '-' + day_data['Unique Label'].astype(str)

        # Convert to a list if needed
        labels = combined_labels.tolist()

        # Draw the labels on the image using the centroids and labels
        updated_labeled_image = draw_labels_on_cells(image, centroids, labels)
        
        # Resize to 2048x2048 if necessary
        updated_labeled_image = updated_labeled_image.resize((2048, 2048))
        
        # Append the updated labeled image to the list
        updated_labeled_images.append(updated_labeled_image)
        
        # Display the updated image
        st.image(updated_labeled_image, caption=f"Updated Image {i + 1} with Cleaned Cell Labels", use_column_width=True)
    
    return updated_labeled_images

import numpy as np
import cv2

@st.cache_data
def cluster_cells_across_days(centroids_list, contours):
    # Assign unique labels to each contour
    unique_labels = np.arange(len(contours))
    
    # Flatten all centroids across days into a single list
    all_centroids = np.vstack(centroids_list)
    
    # Initialize an array for the labels, set to -1 (unlabeled)
    labels = np.full(len(all_centroids), -1)
    
    # Label centroids based on whether they are inside any contour
    for i, contour in enumerate(contours):
        for j, centroid in enumerate(all_centroids):
            # Ensure the centroid is a tuple of (x, y)
            centroid_tuple = (float(centroid[0]), float(centroid[1]))
            
            # Check if the centroid is inside the contour
            if cv2.pointPolygonTest(contour, centroid_tuple, False) >= 0:
                labels[j] = unique_labels[i]
    # For all centroids still labeled -1, assign them to the nearest contour if within 50 pixels
    for j, centroid in enumerate(all_centroids):
        if labels[j] == -1:  # Only process unlabeled centroids
            centroid_tuple = (float(centroid[0]), float(centroid[1]))
            min_distance = float('inf')
            closest_label = -1
            
            # Check distance to each contour to find the closest one
            for i, contour in enumerate(contours):
                distance = abs(cv2.pointPolygonTest(contour, centroid_tuple, True))  # Get absolute distance
                
                # Update if this contour is the closest one so far
                if distance < min_distance:
                    min_distance = distance
                    closest_label = unique_labels[i]
            
            # Assign the centroid to the closest contour's label
            labels[j] = closest_label
    
    # Reshape the labels to match the centroids' original structure across days
    reshaped_labels = []
    idx = 0
    for centroids in centroids_list:
        reshaped_labels.append(labels[idx:idx + len(centroids)])
        idx += len(centroids)
    
    return reshaped_labels
@st.cache_data
# Function to update cell identity based on the given index and new identity values
def update_cell_identity(dataframe, updates):
    # Apply the updates based on matches within the 'Unique Identity' column
    for original_unique_identity, new_identity in updates.items():
        # Find rows where 'Unique Identity' matches the given original index
        mask = dataframe['Unique Identity'] == original_unique_identity
        if mask.any():
            if new_identity.strip() == "":  # Check if the new identity is left blank
                # Remove the row(s) from the dataframe where the mask is True
                dataframe = dataframe[~mask]
                # st.info(f"Cell(s) with Unique Identity {original_unique_identity} have been deleted.")
            else:
                # Update 'Cell Identity', ensuring it's stored as an integer
                dataframe.loc[mask, 'Cell Identity'] = int(new_identity)
        else:
            st.warning(f"Unique Identity {original_unique_identity} not found in the DataFrame.")
    
    # Recalculate the Unique Identity to reflect changes
    dataframe = add_unique_index(dataframe)
    
    # Ensure 'Cell Identity' is of integer type
    dataframe['Cell Identity'] = dataframe['Cell Identity'].astype(int)
    
    return dataframe



# Function to download all labeled images as a zip file in TIFF format
def download_all_images_as_tiff_zip(images, filenames, title, zip_filename="labeled_images_tiff.zip"):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for i, img in enumerate(images):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="TIFF")  # Save image in TIFF format
            zip_file.writestr(filenames[i].replace(".png", ".tiff"), img_buffer.getvalue())  # Change extension to .tiff
    
    # Create a download button for the zip file
    st.download_button(
        label=f"Download All {title} as TIFF ZIP",
        data=zip_buffer.getvalue(),
        file_name=zip_filename,
        mime="application/zip"
    )

# def download_all_binary_images_as_tiff_zip(images, filenames, zip_filename="binary_images_tiff.zip"):
#     zip_buffer = io.BytesIO()
#     with zipfile.ZipFile(zip_buffer, "w") as zip_file:
#         for i, img in enumerate(images):
#             img_buffer = io.BytesIO()
#             img.save(img_buffer, format="TIFF")  # Save image in TIFF format
#             zip_file.writestr(filenames[i].replace(".png", ".tiff"), img_buffer.getvalue())  # Change extension to .tiff
    
#     # Create a download button for the zip file
#     st.download_button(
#         label="Download All Binary Images as TIFF ZIP",
#         data=zip_buffer.getvalue(),
#         file_name=zip_filename,
#         mime="application/zip"
#     )
@st.cache_data
def normalize_pivot_table(pivot_table_cleaned):
    # Divide each row by the value in the first column of that row
    normalized_pivot_table = pivot_table_cleaned.copy()
    normalized_pivot_table.iloc[:, 0:] = normalized_pivot_table.iloc[:, 0:].div(normalized_pivot_table.iloc[:, 0], axis=0)
    
    return normalized_pivot_table


@st.cache_data
# Function to identify and remove cell identities with low area from both the pivot table and full data
def remove_low_area_cells(full_data, pivot_table):
    # Identify the last day (last column in the pivot table)
    last_day_column = pivot_table.columns[-1]
    
    # Find cell identities with area less than 1000 on the last day
    low_area_cells = pivot_table[pivot_table[last_day_column] < 1000].index.tolist()
    
    if low_area_cells:
        st.write(f"Removing cell identities with an area less than 1000 on the last day: {low_area_cells}")
        
        # Remove these cell identities from the full_data DataFrame
        full_data_cleaned = full_data[~full_data['Cell Identity'].isin(low_area_cells)]
        
        # Recreate the pivot table without these cell identities
        pivot_table_cleaned = full_data_cleaned.pivot_table(
            index='Cell Identity', columns='Image Day', values='Area', aggfunc='sum', fill_value=0
        )
        
        return full_data_cleaned, pivot_table_cleaned
    else:
        st.write("No cell identities with an area less than 1000 on the last day.")
        return full_data, pivot_table
    
# Streamlit app
st.title("Image Uploader and Viewer with Geometric Properties")

# Upload normal images
uploaded_files = st.file_uploader("Choose processed files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

# Upload binary images
binary_files = st.file_uploader("Choose binary files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

# Upload label images
label_images = st.file_uploader("Choose label files", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])


if uploaded_files and binary_files and label_images:
    resized_uploaded_images = resize_to_2048(uploaded_files)
    resized_binary_images = resize_to_2048(binary_files)

    images = []
    for image in label_images:
        image = Image.open(image).resize((2048, 2048))
        images.append(image)

    final_images, centroids_across_days, contours_across_days = overlay_images(resized_uploaded_images, resized_binary_images)
    
    labels_across_days = cluster_cells_across_days(centroids_across_days, contours_across_days[-1])
    for image in final_images:
        st.image(image)
       
    full_data = pd.DataFrame()
    labeled_images = []

    binary_images = [ ]
    for image in final_images:
        binary_image = Image.fromarray(image).resize((2048, 2048))
        binary_images.append(Image.fromarray(image))


    for i, (image, centroids, contours, labels) in enumerate(zip(images, centroids_across_days, contours_across_days, labels_across_days)):
        # Append the properly formatted binary image
        print("i" + str(i))
        print(image.size)
        labeled_image = draw_labels_on_cells(image, centroids, labels)
        st.image(labeled_image, caption=f"Image {i+1} with Cell Labels", use_column_width=True)
        # Resize the labeled image to 2048x2048
        labeled_image = labeled_image.resize((2048, 2048))
        labeled_images.append(labeled_image)
        image_data = compute_geometric_properties(contours, centroids, labels, f"Day {i+1}")
        full_data = pd.concat([full_data, image_data], ignore_index=True)

    # Adding the unique index to distinguish cells with the same identity on the same day
    full_data = add_unique_index(full_data)

    st.write("All processed images are displayed with labels.")
    st.write(full_data)

    full_data = full_data.sort_values(by=['Image Day', 'Cell Identity'])

    st.write("Update Cell Identities")
    indices_to_update = st.text_input("Enter the original indices (comma-separated):", "")
    new_identities = st.text_input("Enter the new cell identities (comma-separated):", "")

        
    
# Display the update button
if st.button("Update"):
    # Process updates only when the button is clicked
    if indices_to_update and new_identities:
        try:
            # Convert input strings into lists
            indices_list = indices_to_update.split(',')
            identities_list = new_identities.split(',')
            
            # Ensure both lists have the same length
            if len(indices_list) == len(identities_list):
                # Create a dictionary mapping original indices to new identities
                updates = dict(zip(indices_list, identities_list))
                
                # Update the DataFrame with the new identities
                updated_full_data = update_cell_identity(full_data.copy(), updates)
                full_data = updated_full_data
                # Display the updated DataFrame
                st.write("Updated Data with New Cell Identities:")
                st.write(updated_full_data)
            else:
                st.write(len(indices_list))
                st.write(len(identities_list))
                st.error("The number of indices and new identities must match.")
        except ValueError:
            st.error("Please ensure all inputs are valid strings.")
    else:
        st.error("Please fill out both fields.")
# Download CSV
download_csv(full_data, "cell_geometric_properties.csv")

# Create the pivot table
pivot_table = full_data.pivot_table(index='Cell Identity', columns='Image Day', values='Area', aggfunc='sum', fill_value=0)
st.write(pivot_table)

# Remove low area cells from both pivot table and full data
full_data_cleaned, pivot_table_cleaned = remove_low_area_cells(full_data, pivot_table)


# Display the cleaned data
st.write("Updated Full Data without low area cells:")
st.write(full_data_cleaned)

st.write("Updated Pivot Table without low area cells:")
st.write(pivot_table_cleaned)

# Normalize the pivot table
normalized_pivot_table = normalize_pivot_table(pivot_table_cleaned)

# Display the normalized pivot table
st.write("Normalized Pivot Table:")
st.write(normalized_pivot_table)

# Assuming the pivot_table has been generated correctly
zero_entries = find_zero_entries(pivot_table_cleaned)

# Display the cell identities and days with zero entries
if zero_entries:
    st.write("Entries in the pivot table that are zero:")
    for cell_identity, day in zero_entries:
        st.write(f"Cell Identity: {cell_identity}, Day: {day}")
else:
    st.write("No entries in the pivot table are zero.")

updated_labeled_images = update_labeled_images_from_cleaned_data(
    images=images,
    full_data_cleaned=full_data_cleaned
)


download_area_table(pivot_table_cleaned, 'area_table.csv')

download_area_table(normalized_pivot_table, 'normalized_area_table.csv')
# # # Download all labeled images as a TIFF ZIP
download_all_images_as_tiff_zip(updated_labeled_images, [f"labeled_image_{i + 1}.tiff" for i in range(len(labeled_images))], 'updated labeled images')

#     # Download all labeled images as a TIFF ZIP
download_all_images_as_tiff_zip(binary_images, [f"binary_image_{i + 1}.tiff" for i in range(len(labeled_images))], "binary images")