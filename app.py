import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from PIL import Image
import requests
from io import BytesIO

# Spotify API credentials
CLIENT_ID = "cedd5e4b2b6f4b988bb868ae44cfc454"
CLIENT_SECRET = "5fcabe29b61a4c34b5f105b931bcf404"

# Authenticate with Spotify
sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET
))

# Load the dataset
df = pd.read_csv('song_df.csv')

# Define the features for recommendation
features = ["danceability", "energy", "tempo", "acousticness", "liveness", "speechiness", "instrumentalness"]

# Standardize features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# Title of the app
st.title("Spotify Music Recommendation")

# Display available songs to the user
st.sidebar.header("Choose a song to get recommendations")
song_list = df['name'].unique()
selected_song = st.sidebar.selectbox("Select a Song", song_list)


# Function to fetch Spotify song data
def fetch_song_data(song_name):
    try:
        result = sp.search(q=song_name, type="track", limit=1)
        if result['tracks']['items']:
            song_data = result['tracks']['items'][0]
            album_image = song_data['album']['images'][0]['url']
            spotify_url = song_data['external_urls']['spotify']
            return album_image, spotify_url
    except Exception as e:
        st.error(f"Error fetching data for {song_name}: {e}")
    return None, None


# Function to resize images
def fetch_and_resize_image(url, height=60):
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # Calculate the new width to maintain aspect ratio
        aspect_ratio = img.width / img.height
        new_width = int(height * aspect_ratio)

        # Resize the image
        img_resized = img.resize((new_width, height))
        return img_resized
    except Exception as e:
        st.write(f"Error resizing image: {e}")
        return None


# Function to recommend songs
def recommend_songs(selected_song, df, df_scaled, num_recommendations=20):
    song_index = df[df['name'] == selected_song].index[0]
    song_features = df_scaled.iloc[song_index].values.reshape(1, -1)

    similarity = cosine_similarity(song_features, df_scaled)[0]  # Calculate similarity

    # Get similar songs (excluding the input song)
    similarity_scores = list(enumerate(similarity))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_songs = [df.iloc[i[0]] for i in similarity_scores[1:num_recommendations + 1]]

    return recommended_songs


# Get recommendations when the button is clicked
if st.sidebar.button("Get Recommendations"):
    recommended_songs = recommend_songs(selected_song, df, df_scaled, num_recommendations=20)

    st.header("Recommended Songs")
    for i, song in enumerate(recommended_songs, start=1):
        st.text(f"{i}. {song['name']} ({song['year']}) by {song['artists']}")

        # Fetch Spotify song data (image and link)
        image_url, spotify_url = fetch_song_data(song['name'])
        if image_url:
            resized_image = fetch_and_resize_image(image_url, height=200)
            if resized_image:
                st.image(resized_image)
        else:
            st.write("Image not available")

        # Display the Spotify song link
        if spotify_url:
            st.markdown(f"[Listen on Spotify]({spotify_url})", unsafe_allow_html=True)
        else:
            st.write("Spotify link not available.")

        st.write("---")  # Divider between songs

# Display some general info about the app
st.sidebar.subheader("About")
st.sidebar.text("© 2025 Spotify Music Recommendation. All rights reserved.")
