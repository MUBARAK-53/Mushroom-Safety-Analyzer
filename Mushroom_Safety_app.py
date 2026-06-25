import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="AI Mushroom Safety Analyzer",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# CUSTOM CSS
# -----------------------
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] {
    color: #e5e7eb;
    font-size: 18px;
}

/* Main app background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #064e3b 100%);
}

/* Header */
[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

/* Toolbar */
[data-testid="stToolbar"] {
    right: 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Headings */
h1, h2, h3, h4 {
    color: #f8fafc !important;
}

/* Paragraphs and labels */
p, label, span, div {
    color: #e5e7eb;
}

/* Hero card */
.hero-card {
    background: linear-gradient(135deg, rgba(20,184,166,0.25), rgba(15,23,42,0.85));
    border: 1px solid rgba(255,255,255,0.10);
    padding: 1.8rem;
    border-radius: 22px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    margin-bottom: 1.2rem;
}

/* Info card */
.info-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 1rem 1.2rem;
    border-radius: 16px;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}

/* Selectbox label */
div[class*="stSelectbox"] > label > div[data-testid="stMarkdownContainer"] > p {
    font-size: 20px !important;
    font-weight: 700 !important;
    color: #f8fafc !important;
}

/* Selectbox outer box */
div[data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.88) !important;
    border: 1px solid #14b8a6 !important;
    border-radius: 12px !important;
    min-height: 54px !important;
    color: #ffffff !important;
    font-size: 18px !important;
}

/* Selected value */
div[data-baseweb="select"] span {
    color: #ffffff !important;
    font-size: 18px !important;
    font-weight: 500 !important;
}

/* Dropdown menu */
div[role="listbox"] {
    background-color: #111827 !important;
    border: 1px solid #14b8a6 !important;
    border-radius: 10px !important;
}

/* Dropdown options */
div[role="option"] {
    background-color: #111827 !important;
    color: #f8fafc !important;
    font-size: 18px !important;
}

/* Dropdown hover */
div[role="option"]:hover {
    background-color: #0f766e !important;
    color: #ffffff !important;
}

/* Buttons */
.stButton > button {
    width: 100%;
    min-height: 54px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(90deg, #14b8a6, #0f766e);
    color: white !important;
    font-size: 19px;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(0,0,0,0.22);
}

/* Metrics */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 14px;
    border-radius: 16px;
}

/* Expander */
details {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 0.5rem 0.75rem;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-size: 17px !important;
    font-weight: 600 !important;
}

/* Prediction cards */
.result-safe {
    background: #bbf7d0;
    color: #14532d !important;
    border: 1px solid #86efac;
    padding: 18px;
    border-radius: 16px;
    font-size: 22px;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}

.result-danger {
    background: #fecaca;
    color: #7f1d1d !important;
    border: 1px solid #fca5a5;
    padding: 18px;
    border-radius: 16px;
    font-size: 22px;
    font-weight: 700;
    box-shadow: 0 4px 14px rgba(0,0,0,0.18);
}

.result-safe span, .result-safe div, .result-safe p {
    color: #14532d !important;
}

.result-danger span, .result-danger div, .result-danger p {
    color: #7f1d1d !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08);
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL
# -----------------------
model = XGBClassifier()
model.load_model("mushroom_model.json")

columns = pickle.load(open("feature_columns.pkl", "rb"))

# -----------------------
# FEATURE MAPPINGS
# -----------------------
odor_map = {
    "Creosote": "c",
    "Foul": "f",
    "Anise": "l",
    "Musty": "m",
    "No Odor": "n",
    "Pungent": "p",
    "Spicy": "s",
    "Fishy": "y"
}

stalk_root_map = {
    "Bulbous": "b",
    "Club": "c",
    "Equal": "e",
    "Rooted": "r"
}

cap_color_map = {
    "Cinnamon": "c",
    "Red": "e",
    "Gray": "g",
    "Brown": "n",
    "Pink": "p",
    "Green": "r",
    "Purple": "u",
    "White": "w",
    "Yellow": "y"
}

spore_map = {
    "Chocolate": "h",
    "Black": "k",
    "Brown": "n",
    "Orange": "o",
    "Green": "r",
    "Purple": "u",
    "White": "w",
    "Yellow": "y"
}

bruises_map = {
    "Yes": "t",
    "No": "f"
}

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    st.title("🍄 Mushroom Analyzer")
    st.caption("Interactive ML safety prediction app")

    st.markdown("### Project Information")
    st.info("""
**Model:** XGBoost  
**Dataset:** Mushroom Classification  
**Purpose:** Educational toxicity prediction
""")

    st.markdown("### Safety Warning")
    st.warning(
        "Do not use this app alone to decide whether a wild mushroom is safe to eat."
    )

    st.markdown("### Features Used")
    st.write("- Odor")
    st.write("- Stalk root type")
    st.write("- Bruises")
    st.write("- Cap color")
    st.write("- Spore print color")

# -----------------------
# HEADER
# -----------------------
st.markdown("""
<div class="hero-card">
    <h1>🍄 AI Mushroom Safety Analyzer</h1>
    <p style="font-size:18px; color:#dbeafe; margin-top:8px;">
        Predict whether a mushroom is <b>Edible</b> or <b>Poisonous</b> using a trained XGBoost model.
        Enter visible characteristics and get an instant prediction with confidence.
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------
# TABS
# -----------------------
tab1, tab2, tab3 = st.tabs(["🔍 Analyze", "📘 Feature Guide", "🧠 Model Info"])

# -----------------------
# TAB 1 - ANALYZE
# -----------------------
with tab1:
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        st.subheader("Enter Mushroom Features")

        with st.container(border=True):
            c1, c2 = st.columns(2)

            with c1:
                odor = st.selectbox("Odor", list(odor_map.keys()), key="odor_box")
                stalk_root = st.selectbox("Stalk Root Type", list(stalk_root_map.keys()), key="root_box")
                bruises = st.selectbox("Bruises Present?", list(bruises_map.keys()), key="bruise_box")

            with c2:
                cap_color = st.selectbox("Cap Color", list(cap_color_map.keys()), key="cap_box")
                spore_color = st.selectbox("Spore Print Color", list(spore_map.keys()), key="spore_box")

        analyze = st.button("🔍 Analyze Mushroom")

    with right:
        st.subheader("Selected Traits")
        st.markdown(f"""
        <div class="info-card">
            <p><b>Odor:</b> {odor}</p>
            <p><b>Stalk Root:</b> {stalk_root}</p>
            <p><b>Bruises:</b> {bruises}</p>
            <p><b>Cap Color:</b> {cap_color}</p>
            <p><b>Spore Print:</b> {spore_color}</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Why these features matter")
        st.markdown("""
        <div class="info-card">
            These visible mushroom traits are highly useful for classification in the dataset.
            Odor and spore print color are especially important, while bruises, cap color,
            and stalk root structure add more predictive signal.
        </div>
        """, unsafe_allow_html=True)

    if analyze:
        sample = pd.DataFrame(0, index=[0], columns=columns)

        selected_columns = [
            f"odor_{odor_map[odor]}",
            f"stalk-root_{stalk_root_map[stalk_root]}",
            f"bruises_{bruises_map[bruises]}",
            f"cap-color_{cap_color_map[cap_color]}",
            f"spore-print-color_{spore_map[spore_color]}"
        ]

        for col in selected_columns:
            if col in sample.columns:
                sample[col] = 1

        pred = model.predict(sample)[0]
        probs = model.predict_proba(sample)[0]
        confidence = probs.max()
        poison_risk = probs[1] if len(probs) > 1 else confidence

        st.divider()
        st.subheader("Prediction Result")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric("Confidence", f"{confidence:.2%}")

        with m2:
            st.metric("Prediction", "Edible" if pred == 0 else "Poisonous")

        with m3:
            st.metric("Poison Risk", f"{poison_risk:.2%}")

        st.progress(float(confidence))

        if pred == 0:
            st.markdown(f"""
            <div class="result-safe">
                ✅ EDIBLE<br>
                <span style="font-size:18px; font-weight:600;">
                    Confidence: {confidence:.2%}
                </span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-danger">
                ☠️ POISONOUS<br>
                <span style="font-size:18px; font-weight:600;">
                    Confidence: {confidence:.2%}
                </span>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Model Explanation")
        st.markdown("""
        <div class="info-card">
            The model uses encoded categorical inputs and compares the feature pattern with what it learned during training.
            Strong indicators in this mushroom dataset often include odor, bruising behavior,
            stalk root structure, and spore print color.
        </div>
        """, unsafe_allow_html=True)

        with st.expander("See encoded input row used for prediction"):
            st.dataframe(sample, use_container_width=True)

# -----------------------
# TAB 2 - FEATURE GUIDE
# -----------------------
with tab2:
    st.subheader("Feature Reference")
    g1, g2 = st.columns(2, gap="large")

    with g1:
        st.markdown("""
### Odor
One of the strongest indicators in mushroom classification.

### Stalk Root
The shape of the stalk base helps distinguish classes.

### Bruises
Color or texture change after pressure can provide useful clues.
""")

    with g2:
        st.markdown("""
### Cap Color
This adds supporting visual information.

### Spore Print Color
A classic mushroom-identification feature often used in classification.
""")

# -----------------------
# TAB 3 - MODEL INFO
# -----------------------
with tab3:
    st.subheader("About the Model")

    st.markdown("""
- **Algorithm:** XGBoost Classifier  
- **Input Type:** One-hot encoded categorical features  
- **Prediction Output:** Edible or Poisonous  
- **Use Case:** Educational machine learning demonstration
""")

    with st.expander("How the prediction works"):
        st.write("""
The selected mushroom traits are converted into a one-hot encoded feature row.
That row is passed into the trained XGBoost model, which returns both a class prediction
and probability scores.
""")

    with st.expander("Important warning"):
        st.error("This app is for educational purposes only, not for real-world food safety decisions.")