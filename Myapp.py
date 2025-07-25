import streamlit as st
import joblib as jb
import requests
from bs4 import BeautifulSoup

# Load the saved model and vectorizer
model = jb.load('product_centricity_model.pkl')
vectorizer = jb.load('tfidf_vectorizer.pkl')

# Function to scrape and clean text from a URL
def get_page_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return text
        else:
            return None
    except Exception as e:
        return None

# Predict product centricity
def predict_product_centricity(url):
    text = get_page_text(url)
    if text:
        features = vectorizer.transform([text])
        score = model.predict(features)[0]
        return round(float(score), 3)
    else:
        return None

# Streamlit app layout
st.title("🛍️ Amazon Ad Challenge – Product Centricity Predictor")
st.write("Enter a URL and find out how product-focused the page is (scale 0 to 1).")

url_input = st.text_input("🔗 Enter URL:", "")

if st.button("Predict"):
    if url_input:
        with st.spinner("Fetching and analyzing the page..."):
            score = predict_product_centricity(url_input)
            if score is not None:
                st.success(f"🎯 Product Centricity Score: **{score}**")
                if score >= 0.7:
                    st.info("✅ This page is highly product-centric.")
                elif score >= 0.3:
                    st.warning("⚠️ This page is moderately product-focused.")
                else:
                    st.error("❌ This page is not product-centric.")
            else:
                st.error("Failed to fetch or analyze the URL.")

