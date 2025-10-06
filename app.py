import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64

# --- Beautiful Team Card Example ---
import base64
from PIL import Image

def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Load local image
nida_img_base64 = get_image_base64("/mnt/data/a9296cd9-a2c5-467e-918d-e5d58b77788c.png")

st.markdown("""
<h3 style="text-align:center; color:#1e3c72; font-family: 'Playfair Display', serif; margin-top:40px;">
    Project Team
</h3>
""", unsafe_allow_html=True)

st.markdown(f"""
<div style="display:flex; justify-content:center; gap:30px; flex-wrap:wrap; margin-top:30px;">
    <div style="background:white; padding:20px; border-radius:15px; width:180px; text-align:center; 
                box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: transform 0.3s;">
        <img src="data:image/png;base64,{nida_img_base64}" style="width:100px; height:100px; border-radius:50%; object-fit:cover; box-shadow: 0 2px 8px rgba(0,0,0,0.2);">
        <h4 style="margin:10px 0 5px 0; color:#2a5298; font-family: 'Playfair Display', serif;">Nida</h4>
        <p style="color:gray; margin:0; font-size:14px;">Team Lead</p>
    </div>
</div>

<style>
div div:hover {
    transform: translateY(-5px);
}
</style>
""", unsafe_allow_html=True)


# --- Page Setup ---
st.set_page_config(page_title="TMJ Symmetry Detection", page_icon="🦷", layout="wide")

# --- Helper to encode image to base64 ---
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_image_base64("Dept logo (2).png")  # Your logo file

import streamlit as st

st.markdown(f"""
 <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap" rel="stylesheet"> 

 <header role="banner" style="
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
 ">
    <!-- Left Logo -->
    <img src="data:image/png;base64,{img_base64}" style="height:120px; margin-right:20px;"> 

    <!-- Center Text -->
    <div style="text-align: center; flex-grow: 1; color: white;"> 
        <h1 style=" margin: 0; font-family: 'Playfair Display', serif; font-size: 36px; 
                    letter-spacing: 2px; text-shadow: 2px 2px 6px rgba(0,0,0,0.5); "> 
            MALNAD COLLEGE OF ENGINEERING
        </h1> 
        <h3 style=" margin: 0; font-family: 'Playfair Display', serif; font-size: 24px; 
                    font-weight: normal; color: #f0f0f0; "> 
            CSE (Artificial Intelligence and Machine Learning) 
        </h3> 
    </div> 

    <!-- Right Logo -->
    <img src="https://qs-igauge.blr1.cdn.digitaloceanspaces.com/Image_1292EDB5_5641_F2C1_41BA_B225C09396B4_en.png" style="height:120px; margin-left:20px;"> 
 </header> 

 <style>
     /* Add padding so content doesn't overlap with fixed header */
     .block-container {{
         padding-top: 180px;
     }}
 </style>
""", unsafe_allow_html=True)





# --- Main Title ---
st.markdown("<h2 style='text-align: center;'>🦷 TMJ Symmetry Detection</h2>", unsafe_allow_html=True)

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

    # Run YOLO Detection
    results = model.predict(source=image, conf=0.25)
    asymmetry_percent, width_diff, height_diff = None, None, None
    processed_image = image.copy()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()

        if len(boxes) != 2:
            st.warning(f"⚠️ Expected 2 objects, but detected {len(boxes)} — please recheck image.")
            continue

        boxes = sorted(boxes, key=lambda b: b[0])  # Sort by x position
        left_box, right_box = boxes[0], boxes[1]

        # Center and Size Calculations
        left_cx = (left_box[0] + left_box[2]) / 2
        right_cx = (right_box[0] + right_box[2]) / 2
        left_w = left_box[2] - left_box[0]
        right_w = right_box[2] - right_box[0]
        left_h = left_box[3] - left_box[1]
        right_h = right_box[3] - right_box[1]

        image_center_x = image.shape[1] / 2
        left_distance = abs(left_cx - image_center_x)
        right_distance = abs(right_cx - image_center_x)

        # Symmetry Calculation
        symmetry_error = abs(left_distance - right_distance)
        max_possible_distance = image_center_x
        asymmetry_percent = (symmetry_error / max_possible_distance) * 100
        width_diff = abs(left_w - right_w) / max(left_w, right_w) * 100
        height_diff = abs(left_h - right_h) / max(left_h, right_h) * 100

        # Draw Boxes
        for box, label in zip([left_box, right_box], ['Left', 'Right']):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed_image, f'{label}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if asymmetry_percent is not None:
        # Show Results
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

        # Show Processed Image
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Processed Image", use_container_width=True)

        # Download Button
        result_pil = Image.fromarray(image_rgb)
        buf = io.BytesIO()
        result_pil.save(buf, format="JPEG")
        byte_im = buf.getvalue()

        st.download_button(
            label="📥 Download Processed Image",
            data=byte_im,
            file_name="tmj_result.jpg",
            mime="image/jpeg"
        )

import streamlit as st
import base64

# --- Helper to encode local images to base64 ---
def get_image_base64(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --- Project Guide Section ---
st.markdown("""
<hr style="border: 1px solid #ccc; margin: 40px 0;">
<h3 style="text-align: center; color: #1e3c72; font-family: 'Playfair Display', serif;">
    Project Guide
</h3>
""", unsafe_allow_html=True)

guide_img_base64 = get_image_base64("WhatsApp Image 2025-10-06 at 9.56.43 PM.jpeg")

st.markdown(f"""
<div style="display:flex; justify-content:center; margin-top:20px;">
    <div style="text-align:center; margin:0 20px;">
        <img src="data:image/jpeg;base64,{guide_img_base64}" width="100" style="border-radius:50%; margin-bottom:10px;">
        <h4 style="margin:0; color:#2a5298;">Prof. [Guide Name]</h4>
        <p style="color:gray; margin:0;">CSE Department, AI & ML</p>
    </div>
</div>
""", unsafe_allow_html=True)


# --- Project Team Section ---
st.markdown("""
<hr style="border: 1px solid #ccc; margin-top:40px;">
""", unsafe_allow_html=True)

st.markdown("<h3 style='text-align:center; color:#1e3c72;'>Project Team</h3>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

# Team Member 1
nida_img = get_image_base64("WhatsApp Image 2025-10-06 at 9.57.14 PM.jpeg")
with col1:
    st.markdown(f"""
    <div style="text-align:center;">
        <img src="data:image/jpeg;base64,{nida_img}" width="100">
        <h4 style="margin-top:10px; color:#2a5298;">Nida</h4>
        <p style="color:gray; margin:0;">Team Lead</p>
    </div>
    """, unsafe_allow_html=True)

# Team Member 2 (External URL image)
with col2:
    st.markdown("""
    <div style="text-align:center;">
        <img src="https://cdn-icons-png.flaticon.com/512/4140/4140048.png" width="100">
        <h4 style="margin-top:10px; color:#2a5298;">Ananya</h4>
        <p style="color:gray; margin:0;">Team Member</p>
    </div>
    """, unsafe_allow_html=True)

# Team Member 3
rahul_img = get_image_base64("WhatsApp Image 2025-10-06 at 10.13.18 PM.jpeg")
with col3:
    st.markdown(f"""
    <div style="text-align:center;">
        <img src="data:image/jpeg;base64,{rahul_img}" width="100">
        <h4 style="margin-top:10px; color:#2a5298;">Rahul</h4>
        <p style="color:gray; margin:0;">Team Member</p>
    </div>
    """, unsafe_allow_html=True)

# Optional: Include guide again in the team section as the 4th column
with col4:
    st.markdown(f"""
    <div style="text-align:center;">
        <img src="data:image/jpeg;base64,{guide_img_base64}" width="100">
        <h4 style="margin-top:10px; color:#2a5298;">Prof. [Guide Name]</h4>
        <p style="color:gray; margin:0;">Project Guide</p>
    </div>
    """, unsafe_allow_html=True)



st.markdown("""
<hr style="border: 1px solid #ccc; margin-top:40px;">
""", unsafe_allow_html=True)
