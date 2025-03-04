import streamlit as st
import requests
import os
API_KEY = os.getenv("NEWSAPI_KEY")
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from fastapi import FastAPI
import uvicorn

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')

# Load API Key from Streamlit Secrets
API_KEY = os.getenv("NEWSAPI_KEY")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# List of biased words (expandable)
BIAS_WORDS = {"propaganda", "obviously", "clearly", "blatantly", "undeniably", "shameful", "reckless", "evil"}

# FastAPI instance
api_app = FastAPI()

# Streamlit UI
st.title("📰 News Bias Analyzer")
st.write("Analyze bias in news articles over time!")

# Function to analyze bias
def analyze_bias(article_text):
    sentiment_score = sia.polarity_scores(article_text)['compound']
    words = set(article_text.lower().split())
    bias_word_count = len(words.intersection(BIAS_WORDS))
    blob = TextBlob(article_text)
    subjectivity = blob.sentiment.subjectivity

    bias_score = 10 - (abs(sentiment_score * 10) + bias_word_count * 1.5 + subjectivity * 5)
    bias_score = max(-10, min(10, bias_score))

    return {
        "Sentiment Score": sentiment_score,
        "Bias Words Detected": bias_word_count,
        "Subjectivity Score": subjectivity,
        "Final Bias Score": bias_score
    }

def fetch_articles(topic, num_articles=5):
    """
    Fetches articles related to a given topic using NewsAPI.
    Displays article titles and sources before analyzing.
    """
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])

    results = []
    for article in articles[:num_articles]:
        if article["content"]:
            results.append({
                "date": article["publishedAt"],
                "title": article["title"],
                "source": article["source"]["name"],
                "url": article["url"],
                "content": article["content"]
            })
    
    return results


# Function to analyze multiple articles over time
def analyze_bias_over_time(articles):
    results = []
    for date, article in articles:
        result = analyze_bias(article)
        result["Date"] = date
        results.append(result)
    df = pd.DataFrame(results)
    df.sort_values(by="Date", inplace=True)
    return df

# Function to plot bias trends
def plot_bias_trends(df):
    plt.figure(figsize=(10,5))
    plt.plot(df["Date"], df["Final Bias Score"], marker='o', linestyle='-', color='b', label='Bias Score')
    plt.axhline(y=0, color='r', linestyle='--', label='Neutral Line')
    plt.xlabel("Date")
    plt.ylabel("Bias Score (-10 to +10)")
    plt.title("Bias Trend Over Time")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid()
    st.pyplot(plt)

# Streamlit UI Components
topic = st.text_input("Enter a news topic:", "Gaza War")
num_articles = st.slider("Number of articles to analyze:", 1, 10, 5)

if st.button("Analyze Bias"):
    articles = fetch_articles(topic, num_articles)

    # Display which articles are being analyzed
    st.subheader("📌 Articles Being Analyzed:")
    for article in articles:
        st.write(f"**{article['title']}** - {article['source']} ([Read more]({article['url']}))")
    
    # ✅ Fix: Convert list of dictionaries to list of tuples (date, content)
    article_data = [(art["date"], art["content"]) for art in articles]

    # Analyze and display results
    result_df = analyze_bias_over_time(article_data)
    st.write(result_df)
    plot_bias_trends(result_df)


# FastAPI Endpoint for API Users
@api_app.get("/analyze/")
def analyze_api(article_text: str):
    return analyze_bias(article_text)

# Run API if executed as main
if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=8000)
