import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

st.set_page_config(page_title="Movie Recommender System", layout="wide")

st.title("Your Movie Recommendations")
st.markdown("Discover movies you'll love based on your preferences!")

def fetch_poster(movie_id):
     url = "https://api.themoviedb.org/3/movie/{}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US".format(movie_id)
     data=requests.get(url)
     data=data.json()
     poster_path = data['poster_path']
     full_path = "https://image.tmdb.org/t/p/w500/"+poster_path
     return full_path

imageUrls = [
    fetch_poster(1632),
    fetch_poster(299536),
    fetch_poster(17455),
    fetch_poster(2830),
    fetch_poster(429422),
    fetch_poster(9722),
    fetch_poster(13972),
    fetch_poster(240),
    fetch_poster(155),
    fetch_poster(598),
    fetch_poster(914),
    fetch_poster(255709),
    fetch_poster(572154)
    ]

st.subheader("Popular Movies")
popular_movie_ids = [1632, 299536, 17455, 2830, 429422, 9722]

cols = st.columns(6)
for i, movie_id in enumerate(popular_movie_ids):
    with cols[i]:
        poster_url = fetch_poster(movie_id)
        if poster_url:
            st.image(poster_url, use_container_width=True)

movies = pd.read_csv('dataset.csv')
movies = movies[['id', 'title', 'overview', 'genre']]
movies['tags'] = movies['overview'].fillna('') + ' ' + movies['genre'].fillna('')
new_data = movies.drop(columns=['overview', 'genre'])  
movies_list=movies['title'].values

def load_and_process_data():
    cv = CountVectorizer(max_features=10000, stop_words='english')
    vector = cv.fit_transform(new_data['tags'].values.astype('U')).toarray() 
    similarity = cosine_similarity(vector)   

    return new_data, similarity
    


new_data, similarity = load_and_process_data()
selectvalue = st.selectbox("Select movie from dropdown", movies_list)

def recommend(movie):
    index=new_data[new_data['title']==movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector:vector[1])
    recommend_movie=[]
    recommend_poster=[]
    for i in distance[1:6]:
        movies_id=new_data.iloc[i[0]].id
        recommend_movie.append(new_data.iloc[i[0]].title)
        recommend_poster.append(fetch_poster(movies_id))
    return recommend_movie, recommend_poster



if new_data is not None:
    
    movie_titles = sorted(new_data['title'].unique())
    
    
    
    if st.button("Get Recommendations", type="primary"):

        movie_name, movie_poster = recommend(selectvalue)
        col1,col2,col3,col4,col5=st.columns(5) 
        with col1:
            st.text(movie_name[0])
            st.image(movie_poster[0])
        with col2:
            st.text(movie_name[1])
            st.image(movie_poster[1])
        with col3:
            st.text(movie_name[2])
            st.image(movie_poster[2])
        with col4:
            st.text(movie_name[3])
            st.image(movie_poster[3])
        with col5:
            st.text(movie_name[4])
            st.image(movie_poster[4])
    
    if len(new_data) > 0:
        sample_movies = new_data['title'].head(10).tolist()
    
