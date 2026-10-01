import os
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# FILE NAMES
# ============================================================

DATASET_FILE = "smart_campus_safety_dataset.csv"
MODEL_FILE = "smart_campus_model.pkl"
LOG_FILE = "campus_incident_log.csv"


# ============================================================
# CAMPUS LOCATIONS
# ============================================================

LOCATIONS = [
    "Main Gate",
    "Parking Area",
    "Hostel",
    "Classroom Block",
    "Laboratory",
    "Library",
    "Canteen",
    "Playground",
    "Corridor",
    "Campus Road"
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Smart Campus Safety",
    page_icon="🚨",
    layout="wide"
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: bold;
        text-align: center;
    }

    .sub-title {
        text-align: center;
        font-size: 18px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATASET CREATION
# ============================================================

def create_dataset():

    np.random.seed(42)

    number_of_records = 2000

    data = {

        "Location": np.random.choice(
            LOCATIONS,
            number_of_records
        ),

        "Crowd_Level": np.random.randint(
            0, 101,
            number_of_records
        ),

        "Motion_Level": np.random.randint(
            0, 101,
            number_of_records
        ),

        "Noise_Level": np.random.randint(
            0, 101,
            number_of_records
        ),

        "Light_Level": np.random.randint(
            0, 101,
            number_of_records
        ),

        "Time_Hour": np.random.randint(
            0, 24,
            number_of_records
        ),

        "Unauthorized_Access": np.random.randint(
            0, 2,
            number_of_records
        ),

        "Weapon_Detected": np.random.randint(
            0, 2,
            number_of_records
        ),

        "Fire_Detected": np.random.randint(
            0, 2,
            number_of_records
        ),

        "Fall_Detected": np.random.randint(
            0, 2,
            number_of_records
        )
    }

    df = pd.DataFrame(data)


    # ========================================================
    # INCIDENT CLASSIFICATION
    # ========================================================

    def classify_incident(row):

        if row["Fire_Detected"] == 1:

            return "Fire"

        elif row["Weapon_Detected"] == 1:

            return "Weapon"

        elif row["Fall_Detected"] == 1:

            return "Fall"

        elif row["Unauthorized_Access"] == 1:

            return "Unauthorized Access"

        elif (
            row["Crowd_Level"] > 80
            and row["Noise_Level"] > 75
        ):

            return "Crowd Risk"

        elif row["Motion_Level"] > 85:

            return "Suspicious Activity"

        else:

            return "Normal"


    df["Incident_Type"] = df.apply(
        classify_incident,
        axis=1
    )


    # ========================================================
    # RISK CLASSIFICATION
    # ========================================================

    def classify_risk(row):

        if row["Incident_Type"] in [
            "Fire",
            "Weapon"
        ]:

            return "High"

        elif row["Incident_Type"] in [
            "Fall",
            "Unauthorized Access",
            "Crowd Risk",
            "Suspicious Activity"
        ]:

            return "Medium"

        else:

            return "Low"


    df["Risk_Level"] = df.apply(
        classify_risk,
        axis=1
    )


    df.to_csv(
        DATASET_FILE,
        index=False
    )

    return df


# ============================================================
# LOAD DATASET
# ============================================================

if os.path.exists(DATASET_FILE):

    df = pd.read_csv(
        DATASET_FILE
    )

else:

    df = create_dataset()


# ============================================================
# PREPROCESSING
# ============================================================

df_encoded = pd.get_dummies(
    df,
    columns=["Location"],
    dtype=int
)


X = df_encoded.drop(
    columns=[
        "Risk_Level",
        "Incident_Type"
    ]
)


y = df_encoded[
    "Risk_Level"
]


# ============================================================
# TRAIN MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def train_model():

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42,

        stratify=y
    )


    model = RandomForestClassifier(

        n_estimators=150,

        random_state=42
    )


    model.fit(
        X_train,
        y_train
    )


    y_pred = model.predict(
        X_test
    )


    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )


    matrix = confusion_matrix(
        y_test,
        y_pred
    )


    return (
        model,
        accuracy,
        report,
        matrix,
        X_test,
        y_test,
        y_pred
    )


(
    model,
    accuracy,
    report,
    matrix,
    X_test,
    y_test,
    y_pred
) = train_model()


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🏫 AI-Based Smart Campus Safety & Alert System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI-powered campus risk prediction, CCTV monitoring '
    'and incident alert management'
    '</div>',
    unsafe_allow_html=True
)

st.write("")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ System Information")

st.sidebar.metric(
    "Dataset Records",
    len(df)
)

st.sidebar.metric(
    "ML Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.sidebar.write(
    "**Algorithm:** Random Forest"
)

st.sidebar.write(
    "**Language:** Python"
)

st.sidebar.write(
    "**Interface:** Streamlit"
)

st.sidebar.write(
    "**Computer Vision:** OpenCV"
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🤖 AI Prediction",
        "📹 CCTV Monitoring",
        "🚨 Alerts",
        "📊 Dataset",
        "📈 Model Performance"
    ]
)


# ============================================================
# TAB 1
# AI RISK PREDICTION
# ============================================================

with tab1:

    st.header(
        "🤖 AI-Based Campus Risk Prediction"
    )

    st.write(
        "Enter the current campus conditions. "
        "The trained Random Forest model predicts the risk level."
    )


    col1, col2 = st.columns(2)


    # --------------------------------------------------------
    # LEFT SIDE
    # --------------------------------------------------------

    with col1:

        location = st.selectbox(
            "📍 Campus Location",
            LOCATIONS
        )


        crowd_level = st.slider(
            "👥 Crowd Level",
            0,
            100,
            30
        )


        motion_level = st.slider(
            "🏃 Motion Level",
            0,
            100,
            30
        )


        noise_level = st.slider(
            "🔊 Noise Level",
            0,
            100,
            30
        )


        light_level = st.slider(
            "💡 Light Level",
            0,
            100,
            70
        )


    # --------------------------------------------------------
    # RIGHT SIDE
    # --------------------------------------------------------

    with col2:

        time_hour = st.slider(
            "🕐 Time (24-hour)",
            0,
            23,
            12
        )


        unauthorized_access = st.selectbox(

            "🔐 Unauthorized Access",

            [0, 1],

            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )


        weapon_detected = st.selectbox(

            "⚠️ Weapon Detection Flag",

            [0, 1],

            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )


        fire_detected = st.selectbox(

            "🔥 Fire Detection Flag",

            [0, 1],

            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )


        fall_detected = st.selectbox(

            "🚑 Fall Detection Flag",

            [0, 1],

            format_func=lambda x:
                "No" if x == 0 else "Yes"
        )


    st.write("")


    # ========================================================
    # PREDICTION
    # ========================================================

    if st.button(
        "🔍 PREDICT CAMPUS RISK",
        use_container_width=True
    ):


        input_data = pd.DataFrame([{

            "Crowd_Level":
                crowd_level,

            "Motion_Level":
                motion_level,

            "Noise_Level":
                noise_level,

            "Light_Level":
                light_level,

            "Time_Hour":
                time_hour,

            "Unauthorized_Access":
                unauthorized_access,

            "Weapon_Detected":
                weapon_detected,

            "Fire_Detected":
                fire_detected,

            "Fall_Detected":
                fall_detected
        }])


        # ----------------------------------------------------
        # Add Location Columns
        # ----------------------------------------------------

        for column in X.columns:

            if column.startswith(
                "Location_"
            ):

                input_data[column] = 0


        location_column = (
            "Location_" + location
        )


        if location_column in input_data.columns:

            input_data[
                location_column
            ] = 1


        # ----------------------------------------------------
        # Arrange Features
        # ----------------------------------------------------

        input_data = input_data[
            X.columns
        ]


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        probabilities = model.predict_proba(
            input_data
        )[0]


        confidence = max(
            probabilities
        ) * 100


        st.subheader(
            f"Predicted Risk Level: {prediction}"
        )


        st.write(
            f"Model confidence estimate: "
            f"**{confidence:.2f}%**"
        )


        # ----------------------------------------------------
        # Risk Result
        # ----------------------------------------------------

        if prediction == "High":

            st.error(
                "🚨 HIGH RISK ALERT!"
            )

            st.warning(
                "Security personnel should verify "
                "the situation immediately."
            )


        elif prediction == "Medium":

            st.warning(
                "⚠️ MEDIUM RISK!"
            )

            st.info(
                "Security personnel should monitor "
                "the location."
            )


        else:

            st.success(
                "✅ LOW RISK"
            )

            st.info(
                "No immediate risk indicated by the model."
            )


# ============================================================
# TAB 2
# CCTV MONITORING
# ============================================================

with tab2:

    st.header(
        "📹 CCTV Monitoring"
    )

    st.write(
        "Upload a CCTV image or video for testing."
    )


    uploaded_file = st.file_uploader(

        "📁 Upload CCTV File",

        type=[
            "jpg",
            "jpeg",
            "png",
            "mp4",
            "avi",
            "mov"
        ]
    )


    if uploaded_file is not None:

        filename = uploaded_file.name.lower()


        # ====================================================
        # IMAGE
        # ====================================================

        if filename.endswith(
            (".jpg", ".jpeg", ".png")
        ):


            file_bytes = np.asarray(

                bytearray(
                    uploaded_file.read()
                ),

                dtype=np.uint8
            )


            image = cv2.imdecode(

                file_bytes,

                cv2.IMREAD_COLOR
            )


            if image is not None:

                image_rgb = cv2.cvtColor(

                    image,

                    cv2.COLOR_BGR2RGB
                )


                st.image(

                    image_rgb,

                    caption="CCTV Image",

                    use_container_width=True
                )


                st.success(
                    "✅ CCTV image loaded successfully."
                )


            else:

                st.error(
                    "❌ Unable to read image."
                )


        # ====================================================
        # VIDEO
        # ====================================================

        elif filename.endswith(
            (".mp4", ".avi", ".mov")
        ):


            video_path = (
                "uploaded_cctv_video.mp4"
            )


            with open(
                video_path,
                "wb"
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )


            st.video(
                video_path
            )


            if st.button(
                "🔍 Analyze Video Motion"
            ):


                cap = cv2.VideoCapture(
                    video_path
                )


                ret, previous_frame = (
                    cap.read()
                )


                if not ret:

                    st.error(
                        "❌ Unable to read video."
                    )


                else:

                    previous_gray = (
                        cv2.cvtColor(
                            previous_frame,
                            cv2.COLOR_BGR2GRAY
                        )
                    )


                    motion_frames = 0

                    total_frames = 0


                    while True:

                        ret, frame = (
                            cap.read()
                        )


                        if not ret:

                            break


                        gray = cv2.cvtColor(

                            frame,

                            cv2.COLOR_BGR2GRAY
                        )


                        difference = (
                            cv2.absdiff(
                                previous_gray,
                                gray
                            )
                        )


                        _, threshold = (
                            cv2.threshold(
                                difference,
                                25,
                                255,
                                cv2.THRESH_BINARY
                            )
                        )


                        movement_pixels = (
                            cv2.countNonZero(
                                threshold
                            )
                        )


                        if movement_pixels > 5000:

                            motion_frames += 1


                        total_frames += 1

                        previous_gray = gray


                    cap.release()


                    st.write(
                        f"Total frames analyzed: "
                        f"**{total_frames}**"
                    )


                    st.write(
                        f"Frames with motion: "
                        f"**{motion_frames}**"
                    )


                    if motion_frames > 0:

                        st.warning(
                            "⚠️ Motion detected in CCTV video."
                        )

                    else:

                        st.success(
                            "✅ No significant motion detected."
                        )


# ============================================================
# INCIDENT LOG FUNCTION
# ============================================================

def save_incident(

    location,
    incident_type,
    risk_level,
    description

):


    current_time = (
        datetime.now()
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )


    if risk_level == "High":

        alert_status = "URGENT ALERT"


    elif risk_level == "Medium":

        alert_status = "WARNING"


    else:

        alert_status = "NORMAL"


    incident = {

        "Date_Time":
            current_time,

        "Location":
            location,

        "Incident_Type":
            incident_type,

        "Risk_Level":
            risk_level,

        "Description":
            description,

        "Alert_Status":
            alert_status
    }


    if os.path.exists(
        LOG_FILE
    ):


        old_data = pd.read_csv(
            LOG_FILE
        )


        new_data = pd.concat(

            [
                old_data,
                pd.DataFrame(
                    [incident]
                )
            ],

            ignore_index=True
        )


    else:

        new_data = pd.DataFrame(
            [incident]
        )


    new_data.to_csv(
        LOG_FILE,
        index=False
    )


# ============================================================
# TAB 3
# INCIDENT ALERTS
# ============================================================

with tab3:

    st.header(
        "🚨 Campus Incident Alert System"
    )


    col1, col2 = st.columns(2)


    with col1:

        incident_location = st.selectbox(

            "📍 Incident Location",

            LOCATIONS,

            key="incident_location"
        )


        incident_type = st.selectbox(

            "⚠️ Incident Type",

            [
                "Normal",
                "Suspicious Activity",
                "Unauthorized Access",
                "Crowd Risk",
                "Fall",
                "Fire",
                "Weapon"
            ]
        )


    with col2:

       import streamlit as st

student_risk = st.selectbox(
    "Student Risk",
    ["Low", "Medium", "High"]
)

st.write("Selected risk:", student_risk)