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
/* Header */
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
    gap: 40px;
    margin-top: 30px;
    max-width: 1300px;
    margin-left: auto;
    margin-right: auto;
}

.team-card {
    background: white;
    padding: 25px;
    border-radius: 20px;
    flex: 1 1 240px;
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
    width: 160px; 
    height: 160px; 
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

/* Center alignment for Project Guide */
.guide-container {
    display: flex;
    justify-content: center;
    margin-top: 25px;
}

/* Download button styling */
.download-btn-container {
    display: flex;
    justify-content: center;
    margin-top: 25px;
}
div.stDownloadButton > button:first-child {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    color: white;
    border-radius: 12px;
    padding: 12px 32px;
    font-size: 17px;
    font-weight: 600;
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 12px rgba(30, 60, 114, 0.4);
    transition: all 0.3s ease-in-out;
}
div.stDownloadButton > button:first-child:hover {
    background: linear-gradient(135deg, #2a5298, #1e3c72);
    transform: scale(1.08);
    box-shadow: 0 6px 18px rgba(42, 82, 152, 0.6);
}

/* Page Padding */
.block-container {
    padding-top: 200px;
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
    return YOLO("best.pt")

model = load_model()

import os
from PIL import Image
import streamlit as st

# --- About the Disease (reliable Streamlit layout using columns) ---
st.markdown("<hr style='margin:40px 0;'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#1e3c72;'>About the Disease</h3>", unsafe_allow_html=True)

# Local image filename (change to your file). If not found, fallback_url is used.
local_img_path = "tmj_disease_image.jpg"   # <-- replace with your local file name if available
fallback_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRl5OMDN4f6Hiii5mM-sbjBNUtgWvVkin76RQ&s"

# Try to open local image
pil_img = None
if os.path.exists(local_img_path):
    try:
        pil_img = Image.open(local_img_path)
    except Exception:
        pil_img = None

# Create a centered outer container (adds side margins)
outer_cols = st.columns([1, 3, 1])  # left spacer, middle content, right spacer
with outer_cols[1]:
    # Two equal inner columns: left for image, right for description
    left_col, right_col = st.columns([1, 1])

    with left_col:
        if pil_img:
            st.image(pil_img, use_container_width=True)  # ✅ updated parameter
        else:
            st.image(fallback_url, use_container_width=True)  # ✅ updated parameter

    with right_col:
        text_html = """
        <div style="text-align:justify; font-size:16px; line-height:1.7; color:#333;">
            <p><strong>Temporomandibular Joint Disorder (TMJ Disorder)</strong> affects the joint connecting your jawbone to your skull.
            It can cause pain, stiffness, and difficulty in jaw movement. The disorder often arises from injury, arthritis,
            or jaw misalignment, leading to <strong>asymmetry</strong> between the left and right joints.<br> Early detection of asymmetry using AI and image processing (like this project) can help identify
            conditions such as <strong>Temporomandibular Joint Osteoarthritis (TMJOA)</strong> at an earlier stage,
            improving treatment outcomes and preventing long-term complications.</p>
        </div>
        """
        st.markdown(text_html, unsafe_allow_html=True)






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

        # Display Processed Image
        image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Processed Image", use_container_width=True)

        # Prepare Download Button
        buf = io.BytesIO()
        Image.fromarray(image_rgb).save(buf, format="JPEG")
        buf.seek(0)

        st.markdown("<div class='download-btn-container'>", unsafe_allow_html=True)
        st.download_button("📥 Download Processed Image", buf.getvalue(), "tmj_result.jpg", "image/jpeg")
        st.markdown("</div>", unsafe_allow_html=True)

import streamlit as st
from openai import OpenAI  # or use other LLMs if preferred

# --- AI Report Section ---
st.markdown("<hr style='margin:40px 0;'>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center; color:#1e3c72;'>AI-Generated Diagnostic Report</h3>", unsafe_allow_html=True)

# Suppose your previous analysis produced these values:
asymmetry_percent = 12.5  # example
height_diff = 2.3
width_diff = 1.8

# Show results first
st.write(f"**Asymmetry Percentage:** {asymmetry_percent}%")
st.write(f"**Height Difference:** {height_diff} mm")
st.write(f"**Width Difference:** {width_diff} mm")

# Add a button to generate the report
if st.button("🧠 Generate AI Report"):
    with st.spinner("Generating report..."):
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])  # ensure your key is in .streamlit/secrets.toml
        
        # Construct a prompt
        prompt = f"""
        You are an expert medical AI assistant. Based on the following jaw asymmetry analysis,
        write a professional, easy-to-understand report suitable for a clinical setting.
        
        - Asymmetry Percentage: {asymmetry_percent}%
        - Height Difference: {height_diff} mm
        - Width Difference: {width_diff} mm
        
        Include:
        - Summary of findings
        - Possible causes (brief)
        - Clinical significance
        - Recommendations or next steps
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        
        report = response.choices[0].message.content
        st.success("✅ Report Generated Successfully!")
        st.markdown(
            f"<div style='background-color:#f9f9f9; padding:20px; border-radius:10px; line-height:1.6;'>{report}</div>",
            unsafe_allow_html=True
        )


# --- Project Guide Section ---
st.markdown("<hr style='margin:40px 0;'><h3 style='text-align:center; color:#1e3c72;'>Project Guide</h3>", unsafe_allow_html=True)
guide_img = get_image_base64("WhatsApp Image 2025-10-06 at 9.56.43 PM.jpeg")
st.markdown(f"""
<div class='guide-container'>
    <div class='team-card'>
        <img src='https://cdn.pixabay.com/photo/2023/02/18/11/00/icon-7797704_640.png' class='team-img'>
        <h4 class='team-name'>Prof. [Guide Name]</h4>
        <p class='team-role'>Department of CSE(AI & ML)</p>
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
        <h4 class='team-name'>Anusha B. M</h4>
        <p class='team-role'>4MC22CI003</p>
    </div>
    <div class='team-card'>
        <img src='https://cdn-icons-png.flaticon.com/512/4140/4140048.png' class='team-img'>
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
