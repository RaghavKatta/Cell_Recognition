# Cell_Recognition
Image Preprocessing and Analysis for Cell Recognition Projects

## What is it for?
Whenever performing cell recognition and tracking (especially for growth or change purposes), we often have several images and can gain insights between these images. Often the image will be too unclear to see where the border is exactly (for AI and even people), without referencing previous and later images. This program preprocesses the locations, borders, and features of cells and interpolates to find the ones it's missing for easy manual intervention later. 

By performing clustering, we can assign the cells to groups across the days to track a cell across all these days for consistent numbering, which isn't possible in conventional image processing software like ImageJ and Cell Profiler. 

## How to Use it?
The main files are app.py and sobel.py. These programs are meant to be used with streamlit, which can be run locally with "streamlit run file_name.py"

### 1) 
Please first upload all your images in PNG format to Sobel.py. It will take all these images and allow you to export initially labeled data, photos, and binarized photos. 
![image](https://github.com/user-attachments/assets/5bc4675f-346d-4984-adc2-2f87311e57fd)
-Upload Screen
![Screenshot 2024-10-22 125710](https://github.com/user-attachments/assets/9fd69cf8-bcea-44fd-97fe-269249b9648c)
-Example of labeled image 
![image](https://github.com/user-attachments/assets/9d32958a-4679-46b4-b071-84af258a5091)
-Download Buttons (at the bottom)

### 2) 
Then take these labeled images and binary images and open them in ImageJ. You can isolate the interpolations, or false guesses within the labeled image set, and either draw them in manually or use thresholding to determine the border for those cells exactly. Whenever there are erroneous cell borders detected in the original binary image, you can delete those as well. This will allow you to export a new set of images in PNG format, which represents the corrections needed to be made to the binary image. The reason that this correction step is done on the labeled images, instead of directly on the binary images, is that it will be difficult to align all the cells and adjustments, rather than making corrections in place. 

### 3) 
Finally, you have to upload 3 sets of images into app.py. 1) corrections images, 2) binarized images, 3) original unlabeled images. At this point, app.py will combine your corrections and your binary files, and then determine the labels and clusters for each of your cells. This not only tracks when cells combine but also automatically deletes erroneous small errors, using an area threshold. It also allows you to easily track the cells across the days by displaying this within Streamlit. 

![image](https://github.com/user-attachments/assets/9b8f1be4-0eac-4e7a-8a2b-18310924d15d)
-Upload Selections 

Finally, whenever there are errors with the labeling, you can simply type in a comma-separated list of wrong labels, and a comma-separated list of accurate labels to allow for easy manual changes and comparisons across the days. 


Finally, you can download all the information and images with a download button at the end. 
