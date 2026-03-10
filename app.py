import streamlit as st
import pickle
import pandas as pd
import requests
import os
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TMDB_API_KEY = os.getenv("TMDB_API_KEY")


# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# Must be the first Streamlit command
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",  # Use the full screen width
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# BACKGROUND IMAGE
# -----------------------------------------------------------------------------
def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://images.unsplash.com/photo-1489599849927-2ee91cede3ba");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()
def fetch_poster(movie_id):
    try:
        url=f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        data = requests.get(url, timeout=5)
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            full_path= "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Poster"
    except Exception as e:
        st.warning(f"Could not fetch poster: {str(e)}")
        return "https://via.placeholder.com/500x750?text=No+Poster"

def recommend(movie):
    index = movies[movies["title"] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)
    recommended_movies_names =[]
    recommended_movies_posters =[]
    for i in distances[1:6]:
        movie_id = movies.iloc[i[0]].movie_id
       
        recommended_movies_posters.append(fetch_poster(movie_id))
         #fetch poster from api
        recommended_movies_names.append(movies.iloc[i[0]].title)

    return recommended_movies_names, recommended_movies_posters

def load_or_create_data():
    """
    Load pickled data if available, otherwise create from CSV files.
    This ensures the app works even if pickle files are corrupted or missing.
    """
    try:
        # Try to load existing pickle files
        movies = pickle.load(open("movies_dict.pkl", "rb"))
        movies = pd.DataFrame(movies)
        similarity = pickle.load(open("similarity.pkl", "rb"))
        st.info("✅ Loaded data from pickle files")
        return movies, similarity
    except Exception as e:
        st.warning(f"⚠️ Could not load pickle files ({str(e)}), rebuilding from CSV files...")
        st.info("🔄 This may take a few minutes...")
        
        try:
            # Try to read CSV files locally first
            movies_df = pd.read_csv("tmdb_5000_movies.csv")
            credits_df = pd.read_csv("tmdb_5000_credits.csv")
        except FileNotFoundError:
            # If local files not found, download from GitHub
            st.info("📥 Downloading data files from GitHub...")
            movies_url = "https://raw.githubusercontent.com/PavitraKhangaonkar/Cinematch_AI/main/tmdb_5000_movies.csv"
            credits_url = "https://raw.githubusercontent.com/PavitraKhangaonkar/Cinematch_AI/main/tmdb_5000_credits.csv"
            
            movies_df = pd.read_csv(movies_url)
            credits_df = pd.read_csv(credits_url)
            
            # Merge datasets
            movies_df = movies_df.merge(credits_df, on="title")
            movies_df = movies_df[["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
            movies_df.dropna(inplace=True)
            
            # Helper functions with error handling
            def safe_literal_eval(obj):
                try:
                    return ast.literal_eval(obj)
                except (ValueError, SyntaxError):
                    return []
            
            def convert(obj):
                return [i["name"] for i in safe_literal_eval(obj) if isinstance(i, dict) and "name" in i]
            
            def convert_cast(obj):
                cast_list = []
                for i, val in enumerate(safe_literal_eval(obj)):
                    if i < 3 and isinstance(val, dict) and "name" in val:
                        cast_list.append(val["name"])
                return cast_list
            
            def fetch_director(obj):
                crew_list = safe_literal_eval(obj)
                for i in crew_list:
                    if isinstance(i, dict) and i.get("job") == "Director" and "name" in i:
                        return [i["name"]]
                return []
            
            # Process data with progress updates
            st.info("📊 Processing movie data...")
            movies_df["genres"] = movies_df["genres"].apply(convert)
            movies_df["keywords"] = movies_df["keywords"].apply(convert)
            movies_df["cast"] = movies_df["cast"].apply(convert_cast)
            movies_df["crew"] = movies_df["crew"].apply(fetch_director)
            movies_df["overview"] = movies_df["overview"].apply(lambda x: str(x).split())
            
            # Remove spaces
            movies_df["genres"] = movies_df["genres"].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
            movies_df["keywords"] = movies_df["keywords"].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
            movies_df["cast"] = movies_df["cast"].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
            movies_df["crew"] = movies_df["crew"].apply(lambda x: [i.replace(" ", "") for i in x] if isinstance(x, list) else [])
            
            # Create tags
            movies_df["tags"] = (
                movies_df["overview"] + movies_df["genres"] + 
                movies_df["keywords"] + movies_df["cast"] + movies_df["crew"]
            )
            
            new_df = movies_df[["movie_id", "title", "tags"]].copy()
            new_df["tags"] = new_df["tags"].apply(lambda x: " ".join(str(i) for i in x).lower() if isinstance(x, list) else "")
            
            # Vectorization
            st.info("🔢 Creating movie vectors...")
            cv = CountVectorizer(max_features=5000, stop_words="english")
            vectors = cv.fit_transform(new_df["tags"]).toarray()
            
            # Cosine similarity
            st.info("📏 Calculating similarity matrix...")
            similarity = cosine_similarity(vectors)
            
            # Convert to dict for consistency
            movies_dict = new_df.to_dict()
            
            # Save for future use
            try:
                pickle.dump(movies_dict, open("movies_dict.pkl", "wb"))
                pickle.dump(similarity, open("similarity.pkl", "wb"))
                st.success("💾 Data rebuilt and saved to pickle files")
            except Exception as save_error:
                st.warning(f"Could not save pickle files: {str(save_error)}")
            
            return pd.DataFrame(movies_dict), similarity
            
        except Exception as rebuild_error:
            st.error(f"❌ Failed to rebuild data: {str(rebuild_error)}")
            st.error("Please check that the CSV files are accessible and the data is valid.")
            # Return None to prevent crash, will be handled in main code
            return (None, None)

st.header("Movie Recommender System with AI")

# Load or create data
movies, similarity = load_or_create_data()

# Check if data loading failed
if movies is None or similarity is None:
    st.error("❌ Failed to load movie data. Please check your internet connection and try again.")
    st.error("If the problem persists, contact support.")
    st.stop()

movies_list = movies['title'].values


movies_list = movies['title'].values
selected_movie_name = st.selectbox(
    'Type or select a movie from the dropdown',
    movies_list
)

if st.button('show Recommendation '):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie_name)
    
    col1, col2, col3, col4 , col5 = st.columns(5)
    
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])
    
    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])