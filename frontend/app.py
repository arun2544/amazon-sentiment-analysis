import streamlit as st
import requests
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------

#API_URL = "http://127.0.0.1:8000/predict"
#API_URL = "http://api:8000/predict"
API_URL = "http://13.50.243.128:8000/predict"



st.set_page_config(
    page_title="BERT Sentiment Analyzer",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.hero {
    padding: 25px 30px;
    border-radius: 15px;
    background: linear-gradient(135deg, #1e293b, #334155);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    margin-bottom: 5px;
}

.hero p {
    color: #cbd5e1;
    font-size: 17px;
}

.result-positive {
    padding: 20px;
    border-radius: 12px;
    background-color: #dcfce7;
    border: 1px solid #86efac;
}

.result-negative {
    padding: 20px;
    border-radius: 12px;
    background-color: #fee2e2;
    border: 1px solid #fca5a5;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Session State
# -----------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Header
# -----------------------------

st.markdown("""
<div class="hero">

<h1>🤖 BERT Sentiment Analyzer</h1>

<p>
Analyze Amazon product reviews using a fine-tuned
BERT sentiment classification model.
</p>

</div>
""", unsafe_allow_html=True)

# -----------------------------
# Main Layout
# -----------------------------

left, right = st.columns([1.4, 1])

# -----------------------------
# Review Input
# -----------------------------

with left:

    st.markdown(
        '<div class="section-title">📝 Analyze a Review</div>',
        unsafe_allow_html=True
    )

    review = st.text_area(
        "Enter your product review",
        height=180,
        placeholder=(
            "Example: The product quality is excellent, "
            "and it works exactly as described..."
        ),
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        analyze = st.button(
            "🔍 Analyze Sentiment",
            use_container_width=True,
            type="primary"
        )

    with col2:
        clear = st.button(
            "🗑️ Clear",
            use_container_width=True
        )

    if clear:
        st.session_state.history = []
        st.rerun()

# -----------------------------
# Model Information
# -----------------------------

with right:

    st.markdown(
        '<div class="section-title">⚙️ Model Information</div>',
        unsafe_allow_html=True
    )

    st.info("""
**Model:** Fine-tuned BERT

**Task:** Binary Sentiment Classification

**Classes:** Positive / Negative

**Input:** Product Review

**Backend:** FastAPI

**Frontend:** Streamlit
""")

# -----------------------------
# Prediction
# -----------------------------

if analyze:

    if not review.strip():

        st.warning("Please enter a review before analyzing.")

    else:

        with st.spinner("Analyzing review with BERT..."):

            try:

                response = requests.post(
                    API_URL,
                    json={"text": review},
                    timeout=30
                )

                if response.status_code == 200:

                    result = response.json()

                    sentiment = result["sentiment"]
                    confidence = result["confidence"]

                    positive_probability = (
                        confidence
                        if sentiment == "Positive"
                        else 1 - confidence
                    )

                    negative_probability = (
                        confidence
                        if sentiment == "Negative"
                        else 1 - confidence
                    )

                    # Save history
                    st.session_state.history.append({
                        "Review": review,
                        "Sentiment": sentiment,
                        "Confidence": confidence
                    })

                    st.divider()

                    st.markdown(
                        '<div class="section-title">📊 Prediction Result</div>',
                        unsafe_allow_html=True
                    )

                    # Result box
                    if sentiment == "Positive":

                        st.markdown(
                            f"""
                            <div class="result-positive">
                                <h2>😊 Positive</h2>
                                <p>The model classified this review as positive.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    else:

                        st.markdown(
                            f"""
                            <div class="result-negative">
                                <h2>😞 Negative</h2>
                                <p>The model classified this review as negative.</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.write("")

                    # Metrics
                    m1, m2, m3 = st.columns(3)

                    with m1:
                        st.metric(
                            "Prediction",
                            sentiment
                        )

                    with m2:
                        st.metric(
                            "Confidence",
                            f"{confidence * 100:.2f}%"
                        )

                    with m3:
                        st.metric(
                            "Review Length",
                            f"{len(review.split())} words"
                        )

                    st.write("")

                    # Probabilities
                    st.markdown("### 📈 Class Probabilities")

                    p1, p2 = st.columns(2)

                    with p1:

                        st.write(
                            f"😊 **Positive: {positive_probability * 100:.2f}%**"
                        )

                        st.progress(
                            min(positive_probability, 1.0)
                        )

                    with p2:

                        st.write(
                            f"😞 **Negative: {negative_probability * 100:.2f}%**"
                        )

                        st.progress(
                            min(negative_probability, 1.0)
                        )

                    # Review shown back
                    st.markdown("### 💬 Analyzed Review")

                    st.info(review)

                else:

                    st.error(
                        f"FastAPI returned an error: "
                        f"{response.status_code}"
                    )

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Could not connect to FastAPI. "
                    "Make sure the FastAPI server is running."
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ The request timed out. "
                    "Please try again."
                )

# -----------------------------
# Prediction History
# -----------------------------

if st.session_state.history:

    st.divider()

    st.markdown(
        '<div class="section-title">📚 Prediction History</div>',
        unsafe_allow_html=True
    )

    history_df = pd.DataFrame(
        st.session_state.history
    )

    history_df["Confidence"] = (
        history_df["Confidence"] * 100
    ).round(2).astype(str) + "%"

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # Summary
    total = len(st.session_state.history)

    positive_count = sum(
        1 for x in st.session_state.history
        if x["Sentiment"] == "Positive"
    )

    negative_count = total - positive_count

    st.markdown("### 📊 Analysis Summary")

    s1, s2, s3 = st.columns(3)

    with s1:
        st.metric(
            "Total Reviews",
            total
        )

    with s2:
        st.metric(
            "Positive",
            positive_count
        )

    with s3:
        st.metric(
            "Negative",
            negative_count
        )

# -----------------------------
# Footer
# -----------------------------

st.divider()

st.caption(
    "BERT Sentiment Analysis • FastAPI + Streamlit • "
    "Fine-tuned Transformer Model"
)