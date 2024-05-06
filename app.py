# app.py
import cv2
import numpy as np
import streamlit as st
from utility import (
    stackImages,
    reorder,
    biggestContour,
    drawRectangle,
    initializeTrackbars,
    valTrackbars,
)

st.title("Document Scanner with Perspective Transformation")

# Create sliders for Canny edge detection thresholds
threshold1 = st.slider("Threshold1 for Canny", min_value=100, max_value=255, value=150)
threshold2 = st.slider("Threshold2 for Canny", min_value=100, max_value=255, value=200)

# Options for output image sizes
predefined_sizes = {
    "640x480": (640, 480),
    "800x600": (800, 600),
    "1024x768": (1024, 768),
    "1280x720": (1280, 720),
    "1920x1080": (1920, 1080),
}

# Select box for predefined sizes or custom size selection
size_choice = st.selectbox(
    "Choose output image size",
    list(predefined_sizes.keys()) + ["Custom"]
)

# If "Custom" is selected, use sliders to define custom width and height
if size_choice == "Custom":
    custom_width = st.slider("Custom Width", 200, 2000, 640)
    custom_height = st.slider("Custom Height", 200, 2000, 480)
    output_width = custom_width
    output_height = custom_height
else:
    output_width, output_height = predefined_sizes[size_choice]

# File uploader for image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Convert to grayscale and apply Gaussian blur
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    # Apply Canny edge detection with the values from sliders
    imgThreshold = cv2.Canny(imgBlur, threshold1, threshold2)

    # Apply dilation and erosion
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the biggest contour
    biggest, _ = biggestContour(contours)

    if biggest.size != 0:
        biggest = reorder(biggest)

        # Prepare the perspective transformation
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [output_width, 0], [0, output_height], [output_width, output_height]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        # Apply the perspective transformation
        imgWarpColored = cv2.warpPerspective(img, matrix, (output_width, output_height))

        # Convert to grayscale and apply adaptive threshold
        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)

        # Display the original image, processed images, and the warped image
        st.image(
            [img, imgGray, imgThreshold],
            caption=["Original", "Gray", "Canny Threshold"],
            width=150,
        )
        st.image(
            [imgWarpColored, imgAdaptiveThre],
            caption=["Warped Perspective", "Adaptive Threshold"],
            width=150,
        )
    else:
        st.warning("No suitable contour found in the image.")