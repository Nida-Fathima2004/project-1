import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64
import os
import requests
import time


# ================================
# Helper: Encode images to Base64
# ================================
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# ================================
# Streamlit Page Config
# ================================
st.set_page_config(page_title="TMJ Symmetry Detection", page_icon="🦷", layout="wide")

# ================================
# CSS Styling
# ================================
st.markdown("""
<style>
/* Header */
.header {
    position: fixed;
    top: 0; left: 0;
    width: 100%; z-index: 999;
    display: flex; align-items: center; justify-content: space-between;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 48px 28px 22px 28px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
}
.header h1 {
    margin:0; font-family: 'Playfair Display', serif;
    font-size:36px; letter-spacing:2px; color:white;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
}
.header h3 {
    margin:0; font-family: 'Playfair Display', serif;
    font-size:24px; font-weight: normal; color: #f0f0f0;
}

/* Team Cards */
.team-container {
    display: flex; flex-wrap: wrap; justify-content: center;
    gap: 40px; margin-top: 30px; max-width: 1300px;
    margin-left: auto; margin-right: auto;
}
.team-card {
    background: white; padding: 25px; border-radius: 20px;
    flex: 1 1 240px; max-width: 280px; text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    transition: transform 0.3s, box-shadow 0.3s;
}
.team-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}
.team-img {
    width: 160px; height: 160px; border-radius: 50%;
    object-fit: cover; margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.team-name {
    margin: 12px 0 6px 0; color: #2a5298;
    font-family: 'Playfair Display', serif; font-size: 18px;
}
.team-role {
    color: gray; font-size: 15px; margin: 0;
}
.guide-container {
    display: flex; justify-content: center; margin-top: 25px;
}
.download-btn-container {
    display: flex; justify-content: center; margin-top: 25px;
}
div.stDownloadButton > button:first-child {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white; border-radius: 12px;
    padding: 12px 32px; font-size: 17px; font-weight: 600;
    border: none; cursor: pointer;
    box-shadow: 0 4px 12px rgba(30, 60, 114, 0.4);
    transition: all 0.3s ease-in-out;
}
div.stDownloadButton > button:first-child:hover {
    background: linear-gradient(135deg, #2a5298, #1e3c72);
    transform: scale(1.08);
    box-shadow: 0 6px 18px rgba(42, 82, 152, 0.6);
}
.block-container {
    padding-top: 200px;
}
</style>
""", unsafe_allow_html=True)

# ================================
# Header Section
# ================================
college_logo = get_image_base64("Dept logo (2).png")
st.markdown(f"""
<div class='header'>
    <img src='data:image/png;base64,{college_logo}' style='height:120px; margin-right:20px;'>
    <div style='text-align:center; flex-grow:1;'>
        <h1>MALNAD COLLEGE OF ENGINEERING</h1>
        <h3>CSE (Artificial Intelligence and Machine Learning)</h3>
    </div>
    <img src='https://qs-igauge.blr1.cdn.digitaloceanspaces.com/Image_1292EDB5_5641_F2C1_41BA_B225C09396B4_en.png' style='height:120px; margin-left:20px;'>
</div>
""", unsafe_allow_html=True)

# ================================
# Title
# ================================
st.markdown("<h2 style='text-align:center;'>🦷 TMJ Symmetry Detection</h2>", unsafe_allow_html=True)

# ================================
# Load YOLO Model
# ================================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ================================
# About the Disease Section
# ================================
st.markdown("<hr style='margin:40px 0;'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#1e3c72;'>About the Disease</h3>", unsafe_allow_html=True)

local_img_path = "tmj_disease_image.jpg"
fallback_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRl5OMDN4f6Hiii5mM-sbjBNUtgWvVkin76RQ&s"

if os.path.exists(local_img_path):
    try:
        pil_img = Image.open(local_img_path)
    except Exception:
        pil_img = None
else:
    pil_img = None

outer_cols = st.columns([1, 3, 1])
with outer_cols[1]:
    left_col, right_col = st.columns([1, 1])
    with left_col:
        if pil_img:
            st.image(pil_img, use_container_width=True)
        else:
            st.image(fallback_url, use_container_width=True)
    with right_col:
        st.markdown("""
        <div style="text-align:justify; font-size:16px; line-height:1.7; color:#333;">
            <p><strong>Temporomandibular Joint Disorder (TMJ Disorder)</strong> affects the joint connecting your jawbone to your skull.
            It can cause pain, stiffness, and difficulty in jaw movement. The disorder often arises from injury, arthritis,
            or jaw misalignment, leading to <strong>asymmetry</strong> between the left and right joints.<br><br>
            Early detection using AI and image processing can help identify conditions such as 
            <strong>Temporomandibular Joint Osteoarthritis (TMJOA)</strong> early, improving treatment outcomes and preventing long-term complications.</p>
        </div>
        """, unsafe_allow_html=True)

# ================================
# Image Upload + YOLO Detection
# ================================
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    results = model.predict(source=image, conf=0.25)
    processed_image = image.copy()
    asymmetry_percent, width_diff, height_diff = None, None, None

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) != 2:
            st.warning(f"⚠️ Expected 2 objects, but detected {len(boxes)} — please recheck image.")
            continue

        boxes = sorted(boxes, key=lambda b: b[0])
        left_box, right_box = boxes[0], boxes[1]

        left_cx = (left_box[0] + left_box[2]) / 2
        right_cx = (right_box[0] + right_box[2]) / 2
        left_w, right_w = left_box[2]-left_box[0], right_box[2]-right_box[0]
        left_h, right_h = left_box[3]-left_box[1], right_box[3]-right_box[1]

        image_center_x = image.shape[1] / 2
        left_distance = abs(left_cx - image_center_x)
        right_distance = abs(right_cx - image_center_x)

        symmetry_error = abs(left_distance - right_distance)
        asymmetry_percent = (symmetry_error / image_center_x) * 100
        width_diff = abs(left_w - right_w) / max(left_w, right_w) * 100
        height_diff = abs(left_h - right_h) / max(left_h, right_h) * 100

        for box, label in zip([left_box, right_box], ['Left', 'Right']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_image, f'{label}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if asymmetry_percent is not None:
        if asymmetry_percent > 5:
            status_html = f"""
            <div style="background-color:#ffe6e6; padding:15px; border-radius:10px; text-align:center; 
                        border:2px solid #ff4d4d; font-size:18px; font-weight:bold; color:#b30000;">
                ⚠️ Deformation found (TMJOA) <br>
                Asymmetry: {asymmetry_percent:.2f}% <br>
                Width Diff: {width_diff:.2f}% | Height Diff: {height_diff:.2f}%
            </div>
            """
        else:
            status_html = f"""
            <div style="background-color:#e6ffe6; padding:15px; border-radius:10px; text-align:center; 
                        border:2px solid #33cc33; font-size:18px; font-weight:bold; color:#006600;">
                ✅ No deformation <br>
                Asymmetry: {asymmetry_percent:.2f}% <br>
                Width Diff: {width_diff:.2f}% | Height Diff: {height_diff:.2f}%
            </div>
            """
        st.markdown(status_html, unsafe_allow_html=True)

        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Processed Image", use_container_width=True)

        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, format="JPEG")
        buf.seek(0)

        st.markdown("<div class='download-btn-container'>", unsafe_allow_html=True)
        st.download_button("📥 Download Processed Image", buf.getvalue(), "tmj_result.jpg", "image/jpeg")
        st.markdown("</div>", unsafe_allow_html=True)

# ================================
# AI Report Section (Hugging Face)
# ================================
try:
    hf_token = st.secrets["HF_TOKEN"]
except KeyError:
    st.warning("⚠️ Add your Hugging Face API key in Streamlit secrets to enable report generation.")
    hf_token = None

if hf_token:
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {hf_token}"}

    st.markdown("<hr style='margin:40px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; color:#1e3c72;'>🧠 AI-Generated Diagnostic Report</h3>", unsafe_allow_html=True)

    if st.button("🤖 Generate AI Report"):
        if asymmetry_percent is not None:
            prompt = f"""
            You are an AI medical assistant. Based on TMJ asymmetry analysis:
            - Asymmetry: {asymmetry_percent:.2f}%
            - Width Diff: {width_diff:.2f}%
            - Height Diff: {height_diff:.2f}%

            Write a short clinical-style report including:
            - Summary of findings
            - Possible causes
            - Clinical significance
            - Recommendations
            """

            with st.spinner("Generating report..."):
                response = requests.get("YOUR_URL_HERE")

                # ✅ Safe JSON handling
                if response.status_code == 200:
                    try:
                        data = response.json()
                        st.write("✅ Data loaded successfully:", data)
                    except ValueError:
                        st.error("❌ The server did not return valid JSON.")
                        st.text(response.text)  # show raw text for debugging
                else:
                    st.error(f"❌ Request failed with status code: {response.status_code}")
                    st.text(response.text)

                if isinstance(data, list) and "generated_text" in data[0]:
                    report = data[0]["generated_text"]
                    st.success("✅ Report Generated Successfully!")
                    st.markdown(
                        f"<div style='background:#f9f9f9;padding:20px;border-radius:10px;'>{report}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.error("⚠️ Unexpected response format. Try again later.")
        else:
            st.warning("Upload and process an image first.")

# ================================
# Project Team Section
# ================================
st.markdown("<hr style='margin:40px 0;'><h3 style='text-align:center; color:#1e3c72;'>Project Team</h3>", unsafe_allow_html=True)

nida_img = get_image_base64("WhatsApp Image 2025-10-06 at 9.57.14 PM.jpeg")
rahul_img = get_image_base64("WhatsApp Image 2025-10-06 at 10.13.18 PM.jpeg")
guide_img = get_image_base64("WhatsApp Image 2025-10-06 at 9.56.43 PM.jpeg")
keerthana= get_image_base64("WhatsApp Image 2025-10-08 at 4.36.32 PM.jpeg")
guide_html = f"""
<div class='guide-container'>
    <div class='team-card'>
        <img src='https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRkXMn8gfM1nVj_jMD8_dBTb_xY5Utgp2t28Q&s' class='team-img'>
        <h4 class='team-name'>Prof. [Guide Name]</h4>
        <p class='team-role'>CSE Department, AI & ML</p>
    </div>
</div>
"""
st.markdown(guide_html, unsafe_allow_html=True)

team_html = f"""
<div class='team-container'>
    <div class='team-card'>
        <img src='data:image/jpeg;base64,{nida_img}' class='team-img'>
        <h4 class='team-name'>Anusha B. M</h4>
        <p class='team-role'>4MC22CI003</p>
    </div>
    <div class='team-card'>
        <img src='data:image/jpeg;base64,{keerthana}' class='team-img'>
        <h4 class='team-name'>Keerthana H. N.</h4>
        <p class='team-role'>4MC22CI013</p>
    </div>
    <div class='team-card'>
        <img src='data:image/jpeg;base64,{rahul_img}' class='team-img'>
        <h4 class='team-name'>Nida Fathima</h4>
        <p class='team-role'>4MC22CI019</p>
    </div>
    <div class='team-card'>
        <img src='data:image/jpeg;base64,{guide_img}' class='team-img'>
        <h4 class='team-name'>Pratham M. Jain</h4>
        <p class='team-role'>4MC22CI023</p>
    </div>
</div>
"""
st.markdown(team_html, unsafe_allow_html=True)
