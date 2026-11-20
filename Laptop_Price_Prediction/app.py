"""
================================================================
 🔮 LAPTOP PRICE PREDICTOR – Streamlit Web Application
================================================================
 This app loads the trained ML model and allows users to
 predict laptop prices based on their specifications.

 Run with:
     streamlit run app.py
================================================================
"""

import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ──────────────────────────────────────────────────────────
# 📂  Paths
# ──────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "models", "laptop_price_model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "models", "metadata.pkl")


# ──────────────────────────────────────────────────────────
# 🎨  Page Configuration
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="💻 Laptop Price Predictor",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────
# 🧠  Load Model & Metadata  (cached for performance)
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained pipeline and metadata from disk."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    pipe     = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return pipe, metadata


pipe, metadata = load_model()


# ──────────────────────────────────────────────────────────
# 🛑  If model is not found, show instructions
# ──────────────────────────────────────────────────────────
if pipe is None:
    st.error("⚠️  Trained model not found!")
    st.info(
        "Please run the training script first:\n\n"
        "```bash\n"
        "python notebooks/laptop_price_prediction.py\n"
        "```"
    )
    st.stop()


# ──────────────────────────────────────────────────────────
# 🏷️  Header
# ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        color: #2C3E50;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: #7F8C8D;
        font-size: 1.1rem;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 30px;
        text-align: center;
        color: white;
        margin: 20px 0;
    }
    .prediction-price {
        font-size: 3rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .prediction-label {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
    }
    .stButton > button:hover {
        opacity: 0.9;
    }
    .info-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">💻 Laptop Price Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Predict laptop prices using Machine Learning</p>',
    unsafe_allow_html=True,
)

st.markdown("---")


# ──────────────────────────────────────────────────────────
# 🔧  Sidebar – Model Info
# ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ℹ️  Model Info")
    st.markdown(f"**Model:** {metadata['final_model_name']}")
    st.markdown(f"**R² Score:** {metadata['final_r2_score']:.4f}")
    st.markdown("---")
    st.markdown("### 📋 How to Use")
    st.markdown(
        "1. Fill in the laptop specs\n"
        "2. Click **Predict Price**\n"
        "3. View the estimated price"
    )
    st.markdown("---")
    st.markdown(
        "### 🛠️ Built With\n"
        "- Python & Scikit-learn\n"
        "- Streamlit\n"
        "- Ridge Regressor"
    )


# ──────────────────────────────────────────────────────────
# 📝  User Input Form
# ──────────────────────────────────────────────────────────
st.markdown("### 📝 Enter Laptop Specifications")

# ── Row 1: Company & Type ────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    company = st.selectbox(
        "🏢 Brand / Company",
        options=metadata["companies"],
        index=0,
        help="Select the laptop manufacturer.",
    )

with col2:
    type_name = st.selectbox(
        "📦 Laptop Type",
        options=metadata["type_names"],
        index=0,
        help="Select the laptop category.",
    )

# ── Row 2: RAM & Processor ──────────────────────────────
col3, col4 = st.columns(2)

with col3:
    ram = st.selectbox(
        "🧠 RAM (GB)",
        options=metadata["ram_options"],
        index=metadata["ram_options"].index(8) if 8 in metadata["ram_options"] else 0,
        help="Select the RAM size in GB.",
    )

with col4:
    cpu_brand = st.selectbox(
        "⚙️ Processor",
        options=metadata["cpu_brands"],
        index=0,
        help="Select the processor brand / type.",
    )

# ── Row 3: Storage ──────────────────────────────────────
col5, col6 = st.columns(2)

with col5:
    ssd = st.selectbox(
        "💿 SSD Storage (GB)",
        options=[0, 128, 256, 512, 1000],
        index=2,
        help="Select SSD capacity in GB. Use 1000 for 1TB.",
    )

with col6:
    hdd = st.selectbox(
        "💽 HDD Storage (GB)",
        options=[0, 500, 1000, 2000],
        index=0,
        help="Select HDD capacity in GB. Use 1000 for 1TB.",
    )

# ── Row 4: GPU & OS ─────────────────────────────────────
col7, col8 = st.columns(2)

with col7:
    gpu_brand = st.selectbox(
        "🎮 GPU Brand",
        options=metadata["gpu_brands"],
        index=0,
        help="Select the graphics card manufacturer.",
    )

with col8:
    opsys = st.selectbox(
        "🖥️ Operating System",
        options=metadata["os_options"],
        index=0,
        help="Select the operating system.",
    )

# ── Row 5: Screen & Weight ──────────────────────────────
col9, col10 = st.columns(2)

with col9:
    screen_size = st.selectbox(
        "📐 Screen Size (inches)",
        options=[11.6, 13.3, 14.0, 15.6, 17.3],
        index=3,
        help="Select the screen size in inches.",
    )

with col10:
    weight = st.slider(
        "⚖️ Weight (kg)",
        min_value=0.5,
        max_value=4.0,
        value=2.0,
        step=0.1,
        help="Select the laptop weight in kilograms.",
    )

# ── Row 6: Screen Features ──────────────────────────────
col11, col12 = st.columns(2)

with col11:
    touchscreen = st.selectbox(
        "👆 Touchscreen",
        options=["No", "Yes"],
        index=0,
        help="Does the laptop have a touchscreen?",
    )

with col12:
    ips = st.selectbox(
        "🖼️ IPS Display",
        options=["No", "Yes"],
        index=0,
        help="Does the laptop have an IPS display?",
    )

# ── Row 7: Resolution ───────────────────────────────────
resolution = st.selectbox(
    "🔳 Screen Resolution",
    options=[
        "1366x768",
        "1920x1080 (Full HD)",
        "2560x1440 (Quad HD)",
        "2560x1600 (Retina)",
        "3840x2160 (4K Ultra HD)",
    ],
    index=1,
    help="Select the screen resolution.",
)


# ──────────────────────────────────────────────────────────
# 🔮  Prediction
# ──────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔮  Predict Price", use_container_width=True):

    # ── Parse resolution to compute PPI ──────────────────
    res_match = re.search(r"(\d+)x(\d+)", resolution)
    if res_match:
        x_res = int(res_match.group(1))
        y_res = int(res_match.group(2))
    else:
        x_res, y_res = 1920, 1080  # default fallback

    ppi = round(((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size, 2)

    # ── Build the input DataFrame ────────────────────────
    # The column order and names MUST match what the training
    # pipeline expects:
    #   Numeric : Ram, Weight, Touchscreen, IPS, PPI, HDD, SSD
    #   Category: Company, TypeName, Cpu_brand, Gpu_brand, OpSys

    input_data = pd.DataFrame([{
        # Numeric features
        "Ram": ram,
        "Weight": weight,
        "Touchscreen": 1 if touchscreen == "Yes" else 0,
        "IPS": 1 if ips == "Yes" else 0,
        "PPI": ppi,
        "HDD": hdd,
        "SSD": ssd,
        # Categorical features
        "Company": company,
        "TypeName": type_name,
        "Cpu_brand": cpu_brand,
        "Gpu_brand": gpu_brand,
        "OpSys": opsys,
    }])

    # ── Predict (model outputs log-price) ────────────────
    log_price = pipe.predict(input_data)[0]
    predicted_price = int(np.exp(log_price))

    # Round to nearest ₹100
    predicted_price = round(predicted_price / 100) * 100

    # ── Display result ───────────────────────────────────
    st.markdown(
        f"""
        <div class="prediction-box">
            <p class="prediction-label">Estimated Laptop Price</p>
            <p class="prediction-price">₹{predicted_price:,}</p>
            <p class="prediction-label">
                Based on the selected specifications, the estimated
                laptop price is <strong>₹{predicted_price:,}</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Show input summary ───────────────────────────────
    with st.expander("📋 View Input Summary", expanded=False):
        summary_data = {
            "Specification": [
                "Brand", "Type", "RAM", "Processor", "SSD", "HDD",
                "GPU Brand", "OS", "Screen Size", "Weight",
                "Touchscreen", "IPS Display", "Resolution", "PPI",
            ],
            "Selected Value": [
                company, type_name, f"{ram} GB", cpu_brand,
                f"{ssd} GB" if ssd > 0 else "None",
                f"{hdd} GB" if hdd > 0 else "None",
                gpu_brand, opsys, f'{screen_size}"', f"{weight} kg",
                touchscreen, ips, resolution, f"{ppi:.1f}",
            ],
        }
        st.table(pd.DataFrame(summary_data))


# ──────────────────────────────────────────────────────────
# 📝  Footer
# ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #95a5a6; font-size: 0.9rem;">
        💻 Laptop Price Predictor | Built with Streamlit & Scikit-learn<br>
        Machine Learning Project – Regression
    </div>
    """,
    unsafe_allow_html=True,
)
