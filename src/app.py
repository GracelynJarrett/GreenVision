import streamlit as st
import requests
from PIL import Image
import pandas as pd

# ===============================
# CONFIG
# ===============================
API_URL = "http://localhost:8001/predict"

st.set_page_config(page_title="GreenVision", layout="centered")

# Session state for history
if "history" not in st.session_state:
    st.session_state.history = []

# ===============================
# HEADER
# ===============================
st.title("🌿 GreenVision")
st.write("Upload a plant leaf image to detect disease and get treatment recommendations.")

st.caption(
    "This model performs best on clear, centered leaf images. "
    "Performance may drop on real-world images due to dataset differences."
)


st.divider()

# ===============================
# INPUT SECTION
# ===============================
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])
    predict_button = st.button("Predict")

    if uploaded_file and not predict_button:
        st.info("📸 Image uploaded! Click **Predict** to analyze.")
    elif not uploaded_file:
        st.info("⬆️ Upload a clear image of a plant leaf to begin.")

with col2:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Preview", use_container_width=True)

st.divider()

# ===============================
# PREDICTION LOGIC
# ===============================
if predict_button and uploaded_file:

    
    with st.spinner("Running prediction..."):
        try:
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
            }

            response = requests.post(API_URL, files=files)

            
            if response.status_code == 200:
                result = response.json()

                # ✅ SAVE result so it persists across reruns
                st.session_state.result = result

                st.success("Prediction complete ✅")

            
            # ✅ Add to history HERE (inside success block)
                st.session_state.history.append({
                    "Plant": result["plant"],
                    "Disease": result["disease"],
                    "Confidence": f"{result['confidence']*100:.1f}%"
                    })




            else:
                st.error(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            st.error(f"Failed to connect to API: {e}")


    if "result" in st.session_state:

        result = st.session_state.result

    # ===============================
    # RESULTS SECTION
    # ===============================
        st.subheader("🔍 Diagnosis")

        # ✅ Space after header
        st.markdown("")

        # --- Plant & Disease ---
        colA, colB = st.columns(2)

        with colA:
            st.metric("🌿 Plant", st.session_state.result["plant"])

        with colB:
            st.metric("🦠 Disease", st.session_state.result["disease"])

        # ✅ Space between sections
        st.markdown("")

        # --- Treatment ---
        st.markdown("### 💊 Treatment Recommendation")

        st.markdown(
            f"""
            <div style="
                background-color:#f5f5f5;
                padding:15px;
                border-radius:10px;
                color:black;
                font-size:16px;
            ">
                {st.session_state.result["treatment"]}
            </div>
            """,
            unsafe_allow_html=True
            )

        # ✅ Space after treatment
        st.markdown("")

        # --- Status Section ---
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Confidence", f"{st.session_state.result['confidence']*100:.1f}%")
            st.progress(float(st.session_state.result["confidence"]))

        with col2:
            if st.session_state.result["confidence"] < 0.40:
                st.warning("⚠️ Low confidence")
            else:
                st.success("✅ Good confidence")

        with col3:
        
            if "healthy" in st.session_state.result["disease"].lower():
                st.success("✅ This plant is healthy!")
            else:
                st.warning("🦠 Disease detected")



    

        # ✅ Nice divider at the end
        st.divider()


        # ===============================
        # GRAPH SECTION
        # ===============================
        st.subheader("📊 Analysis")

        # ✅ Graph 1 — Confidence Meter
        st.write("### Confidence Level")
        st.progress(float(st.session_state.result["confidence"]))
        st.write(f"Confidence: **{st.session_state.result['confidence']*100:.1f}%**")

        # ✅ Graph 2 — Confidence vs Threshold
        st.write("### Confidence vs Threshold")

        df = pd.DataFrame({
            "Type": ["Confidence", "Threshold"],
            "Value": [st.session_state.result["confidence"], 0.40]
        })

        st.bar_chart(df.set_index("Type"))

        # ✅ Graph 3 — Explanation (VERY important)
        st.caption(
            "Confidence shows how sure the model is about the prediction. "
            "Predictions below 40% are flagged as low confidence."
        )

#===============================
# TOP PLANT TYPES GRAPH
# ===============================
        st.subheader("🌿 Top Plant Types")

        top_preds = result.get("top_predictions", [])

        if top_preds:
            # ✅ Extract labels + confidence correctly
            labels = [p["class"] for p in top_preds]
            confidences = [p["confidence"] for p in top_preds]

            # ✅ Extract plant names
            plants = [label.split("___")[0] for label in labels]

            df_plants = pd.DataFrame({
                "Plant": plants,
                "Confidence": confidences
            })

            # ✅ Group by plant
            df_grouped = df_plants.groupby("Plant").sum()

            st.bar_chart(df_grouped)

# ===============================
# TOP-5 PREDICTIONS GRAPH
# ===============================

        st.subheader("🔝 Top Predictions")

        top_preds = result.get("top_predictions", [])

        if top_preds:
            df_top = pd.DataFrame(top_preds)

            # ✅ Extract clean columns only
            df_top = df_top[["class", "confidence"]]

            # ✅ Convert to percentage
            df_top["confidence"] = df_top["confidence"] * 100

            # ✅ Rename for display
            df_top["class"] = df_top["class"].apply(lambda x: x.replace("___", " - "))

            # ✅ Set index for clean chart
            df_top = df_top.set_index("class")

            # ✅ SINGLE clean series
            st.bar_chart(df_top["confidence"])



# ===============================
# HISTORY SECTION
# ===============================
if st.session_state.history:

    st.subheader("🧠 Prediction History")

    df_history = pd.DataFrame(st.session_state.history)

    st.dataframe(
        df_history,
        use_container_width=True,
        hide_index=True
        )