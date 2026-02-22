import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt

def process_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold to get binary image
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (assuming cell area between 50-5000 pixels, adjust as needed)
    min_area = 50
    max_area = 5000
    cell_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    
    # Calculate sizes and circularity
    sizes = []
    abnormalities = []
    for cnt in cell_contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        sizes.append(area)
        if circularity < 0.7 or area < 100 or area > 4000:  # Adjust thresholds
            abnormalities.append(cnt)
    
    # Draw contours on image
    output_image = image.copy()
    cv2.drawContours(output_image, cell_contours, -1, (0, 255, 0), 2)
    if abnormalities:
        cv2.drawContours(output_image, abnormalities, -1, (255, 0, 0), 2)
    
    # Report
    num_cells = len(cell_contours)
    avg_size = np.mean(sizes) if sizes else 0
    num_abnormal = len(abnormalities)
    
    report = f"""
    Cell Count: {num_cells}
    Average Cell Size (pixels): {avg_size:.2f}
    Abnormal Cells: {num_abnormal}
    """
    
    return output_image, report

st.title("Low-Cost Cell Counter")

uploaded_file = st.file_uploader("Upload a microscope image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Process
    processed_image, report = process_image(image)
    
    # Display
    st.image(processed_image, channels="BGR", caption="Processed Image (Green: Cells, Red: Abnormal)")
    st.text(report)