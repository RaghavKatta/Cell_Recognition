import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json

# Sidebar settings
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("freedraw", "line", "rect", "circle", "transform", "polygon", "point"),
)
stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == "point":
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
else:
    point_display_radius = 0  # Default value if not in point mode
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#ddd")  # Visible background color

# Allow multiple background images to be uploaded
bg_images = st.sidebar.file_uploader("Background images:", type=["png", "jpg"], accept_multiple_files=True)
realtime_update = st.sidebar.checkbox("Update in realtime", True)

# List to store all canvas data for saving
all_canvas_data = []

# Function to scale the path coordinates
def scale_path(path, scale_factor=4):
    scaled_path = []
    for command in path:
        scaled_command = command.copy()
        cmd_type = command[0]
        if cmd_type in ["M", "L"]:  # MoveTo or LineTo commands
            scaled_command[1] *= scale_factor  # x coordinate
            scaled_command[2] *= scale_factor  # y coordinate
        elif cmd_type == "Q":  # QuadraticCurveTo command
            scaled_command[1] *= scale_factor  # x1 coordinate
            scaled_command[2] *= scale_factor  # y1 coordinate
            scaled_command[3] *= scale_factor  # x coordinate
            scaled_command[4] *= scale_factor  # y coordinate
        elif cmd_type == "C":  # BezierCurveTo command
            scaled_command[1] *= scale_factor  # x1 coordinate
            scaled_command[2] *= scale_factor  # y1 coordinate
            scaled_command[3] *= scale_factor  # x2 coordinate
            scaled_command[4] *= scale_factor  # y2 coordinate
            scaled_command[5] *= scale_factor  # x coordinate
            scaled_command[6] *= scale_factor  # y coordinate
        elif cmd_type == "Z":  # ClosePath command, no coordinates to scale
            pass
        else:
            # Handle other command types if necessary
            pass
        scaled_path.append(scaled_command)
    return scaled_path

# Iterate over each uploaded background image and create a canvas for each
if bg_images:
    for idx, bg_image in enumerate(bg_images):
        st.write(f"Canvas {idx+1}:")

        # Open the uploaded image
        bg_image_pil = Image.open(bg_image)

        # Create a fixed 512x512 canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color if not bg_image else None,  # Set background color if no image
            background_image=bg_image_pil if bg_image else None,
            update_streamlit=realtime_update,
            height=1024,
            width=1024,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius,
            display_toolbar=st.sidebar.checkbox(f"Display toolbar for Canvas {idx+1}", True),
            key=f"canvas_{idx+1}",
        )

        # Collect canvas data and scale coordinates
        if canvas_result.json_data is not None:
            # Make a copy of the objects to prevent modifying the original data
            json_objects = canvas_result.json_data["objects"]
            for obj in json_objects:
                # Scale the x and y coordinates
                for attr in ['left', 'top', 'width', 'height']:
                    if attr in obj:
                        obj[attr] *= 4

                # Scale path data if available
                if 'path' in obj:
                    obj['path'] = scale_path(obj['path'], scale_factor=4)

            # Normalize the updated objects into a DataFrame
            objects = pd.json_normalize(json_objects)

            # Convert path data to JSON strings to make them serializable
            if 'path' in objects.columns:
                objects['path'] = objects['path'].apply(json.dumps)

            # Add canvas index and append to the list
            objects['canvas_id'] = idx + 1
            all_canvas_data.append(objects)

            # Display the current canvas results
            if canvas_result.image_data is not None:
                st.image(canvas_result.image_data)

            # Display the DataFrame under each image
            st.write(f"Canvas {idx+1} Data:")
            st.dataframe(objects)
else:
    # Default canvas if no images are uploaded
    st.write("No background images uploaded. Showing a blank canvas.")
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,  # Visible background color
        update_streamlit=realtime_update,
        height=512,  # Fixed canvas size
        width=512,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius,
        display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        key="default_canvas",
    )

    # Collect canvas data and scale coordinates for default canvas
    if canvas_result.json_data is not None:
        # Make a copy of the objects to prevent modifying the original data
        json_objects = canvas_result.json_data["objects"]
        for obj in json_objects:
            # Scale the x and y coordinates
            for attr in ['left', 'top', 'width', 'height']:
                if attr in obj:
                    obj[attr] *= 4

            # Scale path data if available
            if 'path' in obj:
                obj['path'] = scale_path(obj['path'], scale_factor=4)

        # Normalize the updated objects into a DataFrame
        objects = pd.json_normalize(json_objects)

        # Convert path data to JSON strings to make them serializable
        if 'path' in objects.columns:
            objects['path'] = objects['path'].apply(json.dumps)

        objects['canvas_id'] = 'default'
        all_canvas_data.append(objects)

        # Display the canvas results for the default canvas
        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data)

        # Display the DataFrame under the image
        st.write("Default Canvas Data:")
        st.dataframe(objects)

# Button to save all canvas data into a single table
if st.button("Save All Canvas Data"):
    if all_canvas_data:
        # Combine all canvas data into a single DataFrame
        combined_data = pd.concat(all_canvas_data, ignore_index=True)
        st.write("Combined Canvas Data:")
        st.dataframe(combined_data)

        # Optionally, save to CSV or another format
        st.download_button(
            label="Download as CSV",
            data=combined_data.to_csv(index=False),
            file_name="canvas_data.csv",
            mime="text/csv",
        )
    else:
        st.write("No data to save.")
