import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64

# --- Helper to encode images to base64 ---
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Page Config ---
st.set_page_config(page_title="TMJ Symmetry Detection", page_icon="🦷", layout="wide")

# --- CSS for Header, Team Cards, Buttons ---
st.markdown("""
<style>
/* Header (no change) */
.header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    z-index: 999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    padding: 48px 28px 22px 28px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.25);
}
.header h1 {
    margin:0;
    font-family: 'Playfair Display', serif;
    font-size:36px;
    letter-spacing:2px;
    color:white;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.5);
}
.header h3 {
    margin:0;
    font-family: 'Playfair Display', serif;
    font-size:24px;
    font-weight: normal;
    color: #f0f0f0;
}

/* --- Responsive Team Section --- */
.team-container {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 40px;                    /* More spacing between cards */
    margin-top: 30px;
    max-width: 1300px;
    margin-left: auto;
    margin-right: auto;
}

.team-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    flex: 1 1 240px;              /* Slightly larger flexible cards */
    max-width: 280px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    transition: transform 0.3s, box-shadow 0.3s;
}
.team-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.3);
}

.team-img {
    width: 140px; 
    height: 140px; 
    border-radius: 50%; 
    object-fit: cover;
    margin-bottom: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

.team-name {
    margin: 12px 0 6px 0;
    color: #2a5298;
    font-family: 'Playfair Display', serif;
    font-size: 18px;
}
.team-role {
    color: gray;
    font-size: 15px;
    margin: 0;
}

/* Page Padding */
.block-container {
    padding-top: 200px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* Center alignment only for Project Guide section */
.guide-container {
    display: flex;
    justify-content: center;   /* Horizontally center */
    margin-top: 25px;
}
</style>
""", unsafe_allow_html=True)



# --- Header ---
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

# --- Main Title ---
st.markdown("<h2 style='text-align:center;'>🦷 TMJ Symmetry Detection</h2>", unsafe_allow_html=True)

# --- Load YOLO Model ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # Make sure model file is in root folder

model = load_model()

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    results = model.predict(source=image, conf=0.25)
    asymmetry_percent, width_diff, height_diff = None, None, None
    processed_image = image.copy()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        if len(boxes) != 2:
            st.warning(f"⚠️ Expected 2 objects, but detected {len(boxes)} — please recheck image.")
            continue

        boxes = sorted(boxes, key=lambda b: b[0])
        left_box, right_box = boxes[0], boxes[1]

        # Center & size
        left_cx = (left_box[0] + left_box[2]) / 2
        right_cx = (right_box[0] + right_box[2]) / 2
        left_w, right_w = left_box[2]-left_box[0], right_box[2]-right_box[0]
        left_h, right_h = left_box[3]-left_box[1], right_box[3]-right_box[1]

        image_center_x = image.shape[1] / 2
        left_distance = abs(left_cx - image_center_x)
        right_distance = abs(right_cx - image_center_x)

        # Symmetry
        symmetry_error = abs(left_distance - right_distance)
        asymmetry_percent = (symmetry_error / image_center_x) * 100
        width_diff = abs(left_w - right_w) / max(left_w, right_w) * 100
        height_diff = abs(left_h - right_h) / max(left_h, right_h) * 100

        # Draw Boxes
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
        st.download_button("📥 Download Processed Image", buf.getvalue(), "tmj_result.jpg", "image/jpeg")

# --- Project Guide Section ---
st.markdown("<hr style='margin:40px 0;'><h3 style='text-align:center; color:#1e3c72;'>Project Guide</h3>", unsafe_allow_html=True)
guide_img = get_image_base64("WhatsApp Image 2025-10-06 at 9.56.43 PM.jpeg")
st.markdown(f"""
<div class='guide-container'>
    <div class='team-card'>
        <img src='data:image/jpeg;base64,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' class='team-img'>
        <h4 class='team-name'>Prof. [Guide Name]</h4>
        <p class='team-role'>CSE Department, AI & ML</p>
    </div>
</div>

""", unsafe_allow_html=True)

# --- Project Team Section ---
st.markdown("<hr style='margin:40px 0;'><h3 style='text-align:center; color:#1e3c72;'>Project Team</h3>", unsafe_allow_html=True)

nida_img = get_image_base64("WhatsApp Image 2025-10-06 at 9.57.14 PM.jpeg")
rahul_img = get_image_base64("WhatsApp Image 2025-10-06 at 10.13.18 PM.jpeg")

team_html = f"""
<div class='team-container'>
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
"""
st.markdown(team_html, unsafe_allow_html=True)

