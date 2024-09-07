import streamlit as st
import requests
import os
from dotenv import load_dotenv
from transformers import BartForConditionalGeneration, BartTokenizer

# Load environment variables from .env file
#load_dotenv(r'C:\Users\Rishi\Dropbox\My PC (DESKTOP-M51TTDI)\Desktop\projs\News Summerizer\.env')
load_dotenv()
# Fetch API key from environment variable
api_key = os.getenv('NEWS_API_KEY')

# Debugging: Print API key to verify it's loaded
print("API Key:", api_key)

# Load the pre-trained model and tokenizer
model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

# Define function to fetch real-time news
def fetch_news(api_key, category="technology"):
    url = f"https://newsapi.org/v2/top-headlines?category={category}&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    articles = news_data['articles']
    return articles

# Summarize the article using the model
def summarize_article(article_content):
    if article_content:  # Check if the content is not None or empty
        inputs = tokenizer([article_content], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], max_length=60, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return "No content available for summary."

# Streamlit Web App
st.title('Real-Time News Summarizer')

if api_key:
    # Sidebar for category selection
    st.sidebar.header('Select News Category')
    categories = ["technology", "sports", "business", "entertainment", "health", "politics"]
    selected_category = st.sidebar.radio("Categories", categories)

    # Fetch and display news based on selected category
    st.sidebar.subheader('News')
    with st.spinner("Fetching news..."):
        articles = fetch_news(api_key, selected_category)
        
        # Display news articles and summaries with images
        for article in articles:
            st.subheader(article['title'])
            # st.write(article['description'])
            if article.get('urlToImage'):
                st.image(article['urlToImage'], caption='Image', use_column_width=True)
            summary = summarize_article(article.get('content', ''))
            st.write(summary)
            st.write("---")
else:
    st.error("API key is missing. Please set the NEWS_API_KEY environment variable.")
