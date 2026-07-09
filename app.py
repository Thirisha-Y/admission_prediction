import streamlit as st
import numpy as np
import pickle

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

/* Background */
.stApp{
    background: linear-gradient(to right,#eef2f3,#8ec5fc);
}

/* Main Title */
.main-title{
    font-size:42px;
    font-weight:bold;
    text-align:center;
    color:#003366;
}

/* Subtitle */
.sub-title{
    text-align:center;
    font-size:18px;
    color:black;
    margin-bottom:30px;
}

/* ---------- CHANGE WIDGET LABELS TO BLACK ---------- */

/* Streamlit widget labels */
div[data-testid="stWidgetLabel"] p{
    color:#000000 !important;
    font-weight:bold !important;
}

/* Slider labels */
div[data-testid="stSlider"] label p{
    color:#000000 !important;
    font-weight:bold !important;
}

/* Selectbox labels */
div[data-testid="stSelectbox"] label p{
    color:#000000 !important;
    font-weight:bold !important;
}

/* General labels */
label{
    color:#000000 !important;
    font-weight:bold !important;
}

/* Button */
.stButton>button{
    background:#003366;
    color:white;
    border-radius:12px;
    height:55px;
    width:100%;
    font-size:20px;
    font-weight:bold;
    border:none;
}

.stButton>button:hover{
    background:#004c99;
    color:white;
}

/* Card */
.block{
    padding:20px;
    background:white;
    border-radius:15px;
    box-shadow:0px 0px 12px rgba(0,0,0,0.2);
}

/* Prediction Result heading */
h2, h3 {
    color: black !important;
}

/* Metric label (Admission Chance) */
[data-testid="stMetricLabel"] {
    color: black !important;
}

[data-testid="stMetricLabel"] p {
    color: black !important;
    font-weight: bold;
}

/* Metric value (80.72%) */
[data-testid="stMetricValue"] {
    color: black !important;
    font-weight: bold;
}

/* Success message */
[data-testid="stAlert"] {
    color: black !important;
    font-weight: bold;
}

[data-testid="stAlert"] p {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<p class="main-title">🎓 Graduate Admission Prediction</p>',
    unsafe_allow_html=True
)

# Subtitle changed to BLACK
st.markdown(
    '<p style="text-align:center; font-size:18px; color:black; margin-bottom:30px;">Predict your chances of getting admission into your dream university.</p>',
    unsafe_allow_html=True
)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns(2)

with left:

    gre = st.slider("GRE Score",260,340,310)

    toefl = st.slider("TOEFL Score",80,120,105)

    university = st.selectbox(
        "University Rating",
        [1,2,3,4,5]
    )

    sop = st.slider("SOP Strength",1.0,5.0,3.5,0.5)

    lor = st.slider("LOR Strength",1.0,5.0,3.5,0.5)

    cgpa = st.slider("CGPA",6.0,10.0,8.5,0.01)

with right:

    research = st.selectbox(
        "Research Experience",
        [0,1],
        format_func=lambda x:"Yes" if x==1 else "No"
    )

    ielts = st.slider("IELTS Score",5.5,9.0,7.5,0.5)

    internship = st.slider(
        "Internships",
        0,4,1
    )

    work = st.slider(
        "Work Experience (Years)",
        0,5,1
    )

    projects = st.slider(
        "Projects Completed",
        1,7,3
    )

st.write("")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Admission Chance"):

    data = np.array([[
        gre,
        toefl,
        university,
        sop,
        lor,
        cgpa,
        research,
        ielts,
        internship,
        work,
        projects
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]

    prediction = max(0, min(prediction, 1))

    st.markdown("---")

    st.subheader("Prediction Result")

    st.progress(float(prediction))

    st.metric(
        "Admission Chance",
        f"{prediction*100:.2f}%"
    )

    if prediction >= 0.80:

        st.success("🎉 Excellent! You have a very high chance of admission.")

    elif prediction >= 0.60:

        st.info("👍 Good chance! Consider strengthening your profile.")

    elif prediction >= 0.40:

        st.warning("⚠ Moderate chance. Improve GRE, TOEFL or CGPA.")

    else:

        st.error("❌ Low chance. Work on your academic profile before applying.")

st.markdown("---")

st.markdown(
"""
<center>
Made with ❤️ using Streamlit | Machine Learning Project
</center>
""",
unsafe_allow_html=True
)