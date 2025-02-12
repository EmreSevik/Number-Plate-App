import streamlit as st
from PIL import Image
from helper import detect_plate

model_path = "models/plate_detection.pt"

st.title("Plate Recognition Systems 🚗💨")
st.header("Upload an Image")

files = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

if files is not None:
    st.header("Original Image")
    image = Image.open(files).convert('RGB')
    st.image(image, caption='Original Image', use_container_width=True)

    st.header("Detection Results")

    detection_results, cropped_image, is_detected = detect_plate(image, model_path)

    if is_detected != 0:
        st.write("###[INFO].. Plate is detected")

        if cropped_image is not None:
            st.image(cropped_image, caption="Cropped Plate", use_container_width=True)
        else:
            st.write("###[INFO].. Plate detected but cropping failed.")

        st.image(detection_results, caption="Processed Image", use_container_width=True)

    else:
        st.write("###[INFO].. Plate is NOT detected")
        st.image(detection_results, caption="Processed Image", use_container_width=True)
