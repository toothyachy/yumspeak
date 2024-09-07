import streamlit as st
import pandas as pd
import numpy as np
import pickle
import pydeck as pdk
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import json

stop_words = set(stopwords.words('english'))
custom_stopwords = {'good', 'food', 'service', 'great', 'nice', 'delicious', 'restaurant'}
stop_words.update(custom_stopwords)


@st.cache_data
def load_model():
    return Word2Vec.load("models/saved/word2vec.model")


@st.cache_data
def load_place_vectors():
    with open('notebooks/place_vectors.pkl', 'rb') as f:
        place_vectors = pickle.load(f)
    return place_vectors


@st.cache_data
def load_metadata():
    with open('notebooks/restaurant_metadata.json', 'r') as f:
        metadata = [json.loads(line) for line in f]
    return metadata


def recommend_restaurants(user_input, num, place_vectors, wv, stop_words):
    tokenized_input = simple_preprocess(user_input)
    tokenized_input = [w for w in tokenized_input if w not in stop_words]

    input_vectors = [wv[word] for word in tokenized_input if word in wv]

    if input_vectors:
        input_vector = np.mean(input_vectors, axis=0)
    else:
        input_vector = np.zeros(wv.vector_size)

    similarities = {name: cosine_similarity(input_vector.reshape(1, -1), vector.reshape(1, -1))[0][0]
                    for name, vector in place_vectors.items()}

    sorted_similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:num])

    return sorted_similarities


def fetch_metadata(recommendations, metadata):
    unique_recommendations = []
    seen_restaurants = set()

    for entry in metadata:
        if entry['name'] not in seen_restaurants and entry['name'] in recommendations:
            unique_recommendations.append(entry)
            seen_restaurants.add(entry['name'])

    return unique_recommendations


def recommend_nearby_places(user_lat, user_lon, metadata, k=5):
    coords = np.array([[entry['latitude'], entry['longtitude']] for entry in metadata])

    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)

    distances, indices = knn.kneighbors([[user_lat, user_lon]])

    nearby_places = [metadata[idx] for idx in indices[0]]

    return nearby_places


st.title("Restaurant Recommendation System")

word2vec = load_model()
wv = word2vec.wv
place_vectors = load_place_vectors()
metadata = load_metadata()

user_input = st.text_input("Enter your food preference (e.g., 'chicken rice'):")
user_lat = st.number_input("Enter your latitude:", format="%.6f")
user_lon = st.number_input("Enter your longitude:", format="%.6f")

if user_input and user_lat and user_lon:
    recommendations = recommend_restaurants(user_input, 10, place_vectors, wv, stop_words)
    restaurant_metadata = fetch_metadata(recommendations, metadata)
    nearby_places = recommend_nearby_places(user_lat, user_lon, restaurant_metadata, k=5)

    st.write("Nearby Restaurants:")
    for i, entry in enumerate(nearby_places, 1):
        st.write(f"{i}. {entry['name']} (Longitude: {entry['longtitude']}, Latitude: {entry['latitude']})")
        st.write(f"Random Review: {entry['review_text'][:200]}...")


    places_data = pd.DataFrame({
        'lat': [place['latitude'] for place in nearby_places],
        'lon': [place['longtitude'] for place in nearby_places],
        'type': ['restaurant' for _ in nearby_places]
    })


    user_location = pd.DataFrame({
        'lat': [user_lat],
        'lon': [user_lon],
        'type': ['user']
    })


    map_data = pd.concat([user_location, places_data], ignore_index=True)

    user_layer = pdk.Layer( # resutaurant locaion
        'ScatterplotLayer',
        data=user_location,
        get_position='[lon, lat]',
        get_color='[255, 0, 0]',
        get_radius=200,
        pickable=True
    )

    restaurant_layer = pdk.Layer( # user location
        'ScatterplotLayer',
        data=places_data,
        get_position='[lon, lat]',
        get_color='[0, 128, 0]',
        get_radius=100,
        pickable=True
    )

    view_state = pdk.ViewState(
        latitude=user_lat,
        longitude=user_lon,
        zoom=12,
        pitch=50,
    )

    r = pdk.Deck(layers=[user_layer, restaurant_layer], initial_view_state=view_state)
    st.pydeck_chart(r)


import streamlit as st
import pydeck as pdk # library to use
from streamlit_js_eval import get_geolocation

st.title("Find and Visualize Your Location")


location = get_geolocation() # https://github.com/aghasemi/streamlit_js_eval

if location:
    lat = location['coords']['latitude'] # getting geo_loc in real time
    lon = location['coords']['longitude']# getting geo_loc in real time

    st.success(f"Latitude: {lat}, Longitude: {lon}") # just to show if works
    data = [{"lat": float(lat), "lon": float(lon)}]

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position=["lon", "lat"],
        get_radius=10,
        # get_color=[255, 0, 0], #to change cplor
        pickable=True,
    )


    view_state = pdk.ViewState(
        latitude=float(lat),
        longitude=float(lon),
        zoom=15,
        pitch=0,
    )

    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/streets-v11', # map
        initial_view_state=view_state,
        layers=[layer],
    ))
else:
    st.error("Failed to get geolocation. Please ensure your browser allows location access and try again.") #in case of any errpr


import streamlit as st
import joblib
import json
from sentence_transformers import SentenceTransformer, util


with open('data/restaurant_metadata.json') as json_file:
    restaurant_data = json.load(json_file)


review_embeddings = joblib.load('notebooks/review_embeddings.pkl')


model = SentenceTransformer('all-MiniLM-L6-v2')
user_input = st.text_input("Enter your food preference (e.g., sushi, pizza, seafood):")


if user_input:

    input_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embedding, review_embeddings)[0].cpu().numpy()
    top_5_indices = cosine_scores.argsort()[-5:][::-1]


    for i in top_5_indices:
        restaurant = restaurant_data[i]


        st.write(f"Restaurant: {restaurant['name']}")
        st.write(f"Rating: {restaurant.get('main_rating', 'N/A')} | Address: {restaurant.get('address', 'N/A')}")
        st.write(f"Link: {restaurant.get('link', 'N/A')}")
        st.write(f"Location: ({restaurant.get('latitude', 'N/A')}, {restaurant.get('longtitude', 'N/A')})")

        if restaurant['review_text']:
            st.write(f"Review: {restaurant['review_text'][0][:250]}...")
        st.write("---------------------------------------")
