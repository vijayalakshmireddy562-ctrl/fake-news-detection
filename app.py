
import os
import joblib
import streamlit as st
from ddgs import DDGS

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

@st.cache_resource
def load_model():
    if os.path.exists("fake_news_model.pkl"):
        return joblib.load("fake_news_model.pkl")
    return None

st.title("📰 Fake News Detection System")
st.write("Check news predictions and find online sources.")

headline = st.text_input("News headline")
article = st.text_area("Paste the news article here", height=200)

if st.button("Check News"):
    text = (headline + " " + article).strip()

    if not text:
        st.warning("Please enter a headline or article.")
        st.stop()

    # Existing machine-learning prediction
    model = load_model()

    if model is not None:
        prediction = int(model.predict([text])[0])
        if prediction == 1:
            st.write("### ML Prediction: REAL")
        else:
            st.write("### ML Prediction: FAKE")

        st.caption(
            "This is a model prediction, not proof of truth."
        )

    # Search for supporting information online
    st.subheader("🔎 Online Source Check")

    try:
        query = headline.strip() if headline.strip() else text[:200]

        with st.spinner("Searching online sources..."):
            results = DDGS().text(
                query,
                max_results=5
            )

        results = list(results)

        if results:
            st.info(
                "Search results found. Read the sources to "
                "check whether they support the news."
            )

            for item in results:
                st.markdown(f"**{item.get('title', 'Source')}**")
                st.write(item.get("body", ""))
                url = item.get("href", "")

                if url.startswith("https://") or url.startswith("http://"):
                    st.markdown(f"[Open source]({url})")

                st.divider()
        else:
            st.warning(
                "No matching sources found. This does not "
                "mean the news is fake."
            )

    except Exception as e:
        st.warning(
            "Online search is unavailable right now. "
            "Check your internet connection and try again."
        )

    st.subheader("Final Status")
    st.warning(
        "NEEDS VERIFICATION — compare the article with "
        "reliable sources before deciding whether it is true."
    )

