import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64

# ---------------------------
# Helper function to encode images to base64
# ---------------------------
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# ---------------------------
# Page Setup
# ---------------------------
st.set_page_config(page_title="TMJ Symmetry Detection", page_icon="🦷", layout="wide")

# ---------------------------
# CSS Styles
# ---------------------------
st.markdown("""
<style>
/* Team Cards */
.team-card {
    background:white; 
    padding:20px; 
    border-radius:15px; 
    width:180px; 
    text-align:center; 
    box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
    transition: transform 0.3s;
    margin: 10px;
    display: inline-block;
}
.team-card:hover {
    transform: translateY(-5px);
}
.team-img {
    width:100px; 
    height:100px; 
    border-radius:50%; 
    object-fit:cover; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}
.team-name {
    margin: 10px 0 5px 0;
    color: #2a5298;
    font-family: 'Playfair Display', serif;
}
.team-role {
    color: gray;
    font-size: 14px;
    margin: 0;
}
/* Page Padding */
.block-container {
    padding-top: 180px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
college_logo = get_image_base64("Dept logo (2).png")
st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; background: linear-gradient(135deg, #1e3c72, #2a5298); padding:20px; box-shadow:0 4px 12px rgba(0,0,0,0.25);">
    <img src="data:image/png;base64,{college_logo}" style="height:100px;">
    <div style="text-align:center; color:white;">
        <h1 style="margin:0; font-family:Playfair Display;">MALNAD COLLEGE OF ENGINEERING</h1>
        <h3 style="margin:0; font-family:Playfair Display;">CSE (Artificial Intelligence and Machine Learning)</h3>
    </div>
    <img src="https://qs-igauge.blr1.cdn.digitaloceanspaces.com/Image_1292EDB5_5641_F2C1_41BA_B225C09396B4_en.png" style="height:100px;">
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Main Title
# ---------------------------
st.markdown("<h2 style='text-align:center; margin-top:30px;'>🦷 TMJ Symmetry Detection</h2>", unsafe_allow_html=True)

# ---------------------------
# Load YOLO Model
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg","jpeg","png"])
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
        left_box, right_box = boxes

        # Center & Size
        left_cx = (left_box[0] + left_box[2])/2
        right_cx = (right_box[0] + right_box[2])/2
        left_w, right_w = left_box[2]-left_box[0], right_box[2]-right_box[0]
        left_h, right_h = left_box[3]-left_box[1], right_box[3]-right_box[1]

        # Symmetry Calculation
        center_x = image.shape[1]/2
        symmetry_error = abs(abs(left_cx - center_x) - abs(right_cx - center_x))
        asymmetry_percent = (symmetry_error / center_x)*100
        width_diff = abs(left_w - right_w)/max(left_w, right_w)*100
        height_diff = abs(left_h - right_h)/max(left_h, right_h)*100

        # Draw Boxes
        for box, label in zip([left_box, right_box], ['Left','Right']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(processed_image, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(processed_image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    if asymmetry_percent is not None:
        if asymmetry_percent > 5:
            st.markdown(f"<div style='background:#ffe6e6; padding:15px; border-radius:10px; text-align:center; border:2px solid #ff4d4d; color:#b30000;'><b>⚠️ Deformation found (TMJOA)</b><br>Asymmetry: {asymmetry_percent:.2f}%<br>Width Diff: {width_diff:.2f}% | Height Diff: {height_diff:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background:#e6ffe6; padding:15px; border-radius:10px; text-align:center; border:2px solid #33cc33; color:#006600;'><b>✅ No deformation</b><br>Asymmetry: {asymmetry_percent:.2f}%<br>Width Diff: {width_diff:.2f}% | Height Diff: {height_diff:.2f}%</div>", unsafe_allow_html=True)

        # Show Image
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Download
        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)).save(buf, format="JPEG")
        st.download_button("📥 Download Processed Image", buf.getvalue(), "tmj_result.jpg", "image/jpeg")

# ---------------------------
# Project Guide Section
# ---------------------------
st.markdown("<hr><h3 style='text-align:center; color:#1e3c72;'>Project Guide</h3>", unsafe_allow_html=True)
guide_img = get_image_base64("WhatsApp Image 2025-10-06 at 9.56.43 PM.jpeg")
st.markdown(f"""
<div style='text-align:center;'>
    <div class='team-card'>
        <img src='data:image/jpeg;base64,{guide_img}' class='team-img'>
        <h4 class='team-name'>Prof. [Guide Name]</h4>
        <p class='team-role'>CSE Department, AI & ML</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# Team Section
# ---------------------------
st.markdown("<hr><h3 style='text-align:center; color:#1e3c72;'>Project Team</h3>", unsafe_allow_html=True)
nida_img = get_image_base64("WhatsApp Image 2025-10-06 at 9.57.14 PM.jpeg")
rahul_img = get_image_base64("WhatsApp Image 2025-10-06 at 10.13.18 PM.jpeg")

st.markdown(f"""
<div style='text-align:center;'>
    <div class='team-card'>
        <img src='data:image/jpeg;base64,{nida_img}' class='team-img'>
        <h4 class='team-name'>Nida</h4>
        <p class='team-role'>Team Lead</p>
    </div>
    <div class='team-card'>
        <img src='https://cdn-icons-png.flaticon.com/512/4140/4140048.png' class='team-img'>
        <h4 class='team-name'>Ananya</h4>
        <p class='team-role'>Team Member</p>
    </div>
    <div class='team-card'>
        <img src='data:image/jpeg;base64,{rahul_img}' class='team-img'>
        <h4 class='team-name'>Rahul</h4>
        <p class='team-role'>Team Member</p>
    </div>
    <div class='team-card'>
        <img src='data:image/jpeg;base64,{guide_img}' class='team-img'>
        <h4 class='team-name'>Prof. [Guide Name]</h4>
        <p class='team-role'>Project Guide</p>
    </div>
</div>
""", unsafe_allow_html=True)
