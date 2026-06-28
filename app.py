import streamlit as st
import cv2
import numpy as np
import joblib
import gdown
import os

if not os.path.exists("anemia"):
    gdown.download_folder(
        "https://drive.google.com/drive/folders/1sENXyGJ_q4Evqo3cchtMoA9-5b089zEP",
        quiet=False,
        use_cookies=False
    )

if not os.path.exists("normal"):
    gdown.download_folder(
        "https://drive.google.com/drive/folders/1PnTpJ18bfJ6LYH4X9r0BXanfy-UvdsYz",
        quiet=False,
        use_cookies=False
    )
    
from utils.build_dataset import extract_features

model = joblib.load("anemia_model.pkl")

st.title("🩸 Anemia Detection App")
st.write("Upload a blood smear image to analyze, or capture one using your camera.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

camera_image = st.camera_input("Take a picture...")


if camera_image is not None:
    file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Captured Image", width="stretch")

    features = extract_features(img)

    if features is None or features[3] < 20:
        st.error("❌ Not a valid blood smear image. Please upload a microscope image without any unwanted objects.")
        st.stop()

    if features is None:
        st.error("❌ Could not detect cells. Try again.")
        st.stop()

    input_data = [[
        features[0],
        features[1],
        features[2],
        features[3],
        features[4]
    ]]

    prediction = model.predict(input_data)

    st.subheader("🧾 Diagnosis Result")

    if prediction[0] == 0:
        st.success("✅ Likely Normal")
    else:
        st.error("⚠️ Possible Anemia Detected")

    st.subheader("🔬 Cell Features")

    st.write(f"Mean RBC Area: {features[0]:.2f}")
    st.write(f"RBC Size Variation: {features[1]:.2f}")
    st.write(f"Mean Red Intensity: {features[2]:.2f}")
    st.write(f"RBC Count: {features[3]}")
    st.write(f"Red Variation (std): {features[4]:.2f}")
    st.write(f"Pale Ratio: {features[5]:.3f}")


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    features = extract_features(img)

    # 🚨 safety check
    if features is None:
        st.error("❌ No cells detected. Try a clearer image.")
    else:
        input_data = [[
            features[0],
            features[1],
            features[2],
            features[3],
            features[4]
        ]]

        prediction = model.predict(input_data)

        st.subheader("🧾 Diagnosis Result")

        if prediction[0] == 0:
            st.success("✅ Likely Normal")
        else:
            st.error("⚠️ Possible Anemia Detected")

        # ✅ SHOW REAL FEATURES
        st.subheader("🔬 Cell Features")

        st.write(f"Mean RBC Area: {features[0]:.2f}")
        st.write(f"RBC Size Variation: {features[1]:.2f}")
        st.write(f"Mean Red Intensity: {features[2]:.2f}")
        st.write(f"RBC Count: {features[3]}")
