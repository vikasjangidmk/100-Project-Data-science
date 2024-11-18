import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Set page configuration
st.set_page_config(page_title="Book Recommendation System", layout="wide")

# Load the dataset with caching to improve performance
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\vikas\100-Project-Data-science\2 Book recommendation System\books_data.csv")
    data['average_rating'] = pd.to_numeric(data['average_rating'], errors='coerce')
    data.dropna(subset=['title', 'authors'], inplace=True)
    data['book_content'] = data['title'] + ' ' + data['authors']
    return data

data = load_data()

# Sidebar section for user inputs
st.sidebar.title("Book Recommendation System")
st.sidebar.markdown("### Select Book to Get Recommendations")

# Book Recommendation Function
def recommend_books(book_title, cosine_sim, data):
    if book_title not in data['title'].values:
        st.error(f"'{book_title}' not found in the dataset.")
        return []

    # Get the index of the book that matches the title
    idx = data[data['title'] == book_title].index[0]

    # Get the cosine similarity scores for all books with this book
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 most similar books (excluding the input book)
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 recommended books
    return data['title'].iloc[book_indices].tolist()

# Apply TF-IDF Vectorizer and compute cosine similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['book_content'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# User input to select a book
book_title = st.sidebar.selectbox("Choose a Book", data['title'].unique())

if book_title:
    recommended_books = recommend_books(book_title, cosine_sim, data)
    if recommended_books:
        st.subheader(f"Top 10 Books Recommended for '{book_title}'")
        for i, book in enumerate(recommended_books, 1):
            st.write(f"{i}. {book}")

# Footer
st.markdown(""" 
---
**Developed by Vikas**
""")
