import streamlit as st
import geocoder
import pandas as pd
import numpy as np
import json
import pickle
import random
import ast
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from streamlit_js_eval import get_geolocation # https://github.com/aghasemi/streamlit_js_eval
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from yumspeak_ml.params import *


# CONFIG SIZE
st.set_page_config(
    layout="wide",
)
st.markdown(" <style> div[class^='block-container'] { padding-top: 2rem; padding-left: -2rem; padding-right: 0; } </style> ", unsafe_allow_html=True)
st.set_option('deprecation.showPyplotGlobalUse', False)

# GLOBALS
stop_words = set(stopwords.words('english'))
custom_stopwords = set([word[0] for word in STOPWORDS])
stop_words.update(custom_stopwords)

@st.cache_data
def load_metadata():
    with open('data/restaurant_metadata.json', 'r') as f:
        metadata = json.load(f)
    return metadata

@st.cache_data
def load_model():
    return Word2Vec.load("model/word2vec.model")

@st.cache_data
def load_place_vectors():
    with open('model/vectors.pkl', 'rb') as f:
        place_vectors = pickle.load(f)
    return place_vectors

@st.cache_data
def load_tokens():
    with open('model/tokens.pkl', 'rb') as f:
        tokens = pickle.load(f)
    return tokens



# FUNCTIONS
# nearby_places = pd.DataFrame({
#     'name': ['Joo Chiat Banh Mi Ca Phe', 'Banh Mi 233', 'The Viet Roti @ Joo Chiat', 'Nhung Kitchen - Vietnamese Banh Mi', 'Banh Mi Thit by Star Baguette'],
#     'latitude': [1.310403,1.312772,1.310314,1.322767,1.313885],
#     'longitude': [103.901791,103.900092,103.901645,103.851969,103.885366],
#     'similarity': [0.685230,0.680937,0.656748,0.656565,0.653965]
# })


def recommend_restaurants(user_input, num, place_vectors, wv, stop_words):
    tokenized_input = simple_preprocess(user_input)
    tokenized_input = [w for w in tokenized_input if w not in stop_words]
    input_vectors = [wv[word] for word in tokenized_input if word in wv]

    if input_vectors:
        input_vector = np.mean(input_vectors, axis=0)
    else:
        input_vector = np.zeros(wv.vector_size)

    similarities = {}

    for index, values in enumerate(place_vectors):
        similarity = np.dot(input_vector, values) / (np.linalg.norm(input_vector) * np.linalg.norm(values))
        similarities[index] = similarity

    sorted_records = sorted(similarities.items(), key=lambda item: float(item[1]), reverse=True)
    sorted_similarities = {k: v for k, v in sorted_records[:num]}

    return sorted_similarities


def fetch_metadata(recommendations, metadata):
    unique_recommendations = []
    seen_restaurants = set()

    for key, value in recommendations.items():
        entry = metadata[int(key)]
        entry['similarity'] = value
        entry['global_index'] = key
        unique_recommendations.append(entry)

    # for entry in metadata:
    #     if entry['name'] not in seen_restaurants and entry['name'] in recommendations:
    #         unique_recommendations.append(entry)
    #         seen_restaurants.add(entry['name'])

    return unique_recommendations


def recommend_nearby_places(user_lat, user_lon, metadata, k=5):
    coords = np.array([[entry['latitude'], entry['longitude']] for entry in metadata])

    knn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(coords)

    distances, indices = knn.kneighbors([[user_lat, user_lon]])

    nearby_places = [metadata[idx] for idx in indices[0]]

    return nearby_places

def get_latlng(address):
    g = geocoder.mapbox(f"{address},SG", key=st.secrets['mapbox'])
    location = {
        'coords': {
            'latitude': g.json['lat'],
            'longitude': g.json['lng'],
        }}
    return location

def get_wordcloud(index, tokens):
    frequency = Counter(tokens[index])
    word_freq = dict(frequency)
    wordcloud = WordCloud(width=800, height=400, background_color='#F8EDE3', colormap='viridis').generate_from_frequencies(word_freq)
    return wordcloud


# FRONTEND
st.title("YUMSPEAK")
st.subheader("Restaurant Recommender for the Discerning Diner")

word2vec = load_model()
wv = word2vec.wv
place_vectors = load_place_vectors()
tokens = load_tokens()
metadata = load_metadata()
location = get_geolocation()
if location:
    lat = location['coords']['latitude']
    lon = location['coords']['longitude']
nearby_places = None


with st.sidebar:
    with st.form("user_query"):
        user_input = st.text_input("Keywords", value=("banh mi"))
        address = st.text_input("Address", value=("my location"))
        submitted = st.form_submit_button("Submit")

        if submitted:
            recommendations = recommend_restaurants(user_input, 10, place_vectors, wv, stop_words)

            restaurant_metadata = fetch_metadata(recommendations, metadata)

            if address != "my location":
                location = get_latlng(address)
                lat = location['coords']['latitude']
                lon = location['coords']['longitude']
            # st.write(f"User's selected address: {address} and coordinates: {location}")

            nearby_places = recommend_nearby_places(location['coords']['latitude'], location['coords']['longitude'], restaurant_metadata, k=5)


if nearby_places != None:

    # st.subheader(f"Our Recommendations")
    # names = [restaurant['name'] for restaurant in restaurant_metadata]
    # st.write(f'We recommend {", ".join(names)}')

    st.subheader(f"Places Near You")

    # nb = [f"name: {restaurant['name']}, similarity: {restaurant['similarity']}, key: {restaurant['global_index']}" for restaurant in nearby_places]
    # st.write(nb)

    places_data = pd.DataFrame({
        'lat': [place['latitude'] for place in nearby_places],
        'lon': [place['longitude'] for place in nearby_places],
        'type': ['restaurant' for _ in nearby_places]
    })

    user_location = pd.DataFrame({
        'lat': [lat],
        'lon': [lon],
        'type': ['user']
    })

    map_data = pd.concat([user_location, places_data], ignore_index=True)

    user_layer = pdk.Layer( # restaurant locaion
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
        latitude=lat,
        longitude=lon,
        zoom=12,
        pitch=0,
    )

    r = pdk.Deck(map_style='mapbox://styles/mapbox/streets-v11', layers=[user_layer, restaurant_layer], initial_view_state=view_state)
    st.pydeck_chart(r)

    place_names = [f"{i+1}-{v['name'][:20]}.." for i,v in enumerate(nearby_places)]
    tabs = st.tabs(place_names)

    for i, tab in enumerate(tabs):
        with tab:
            st.title(f"{nearby_places[i]['name']} - {nearby_places[i]['main_rating']} ({nearby_places[i]['cuisine']})")
            st.write(f"{nearby_places[i]['address']} [See on Google Map]({nearby_places[i]['link']})")

            st.subheader(f"What Others Say")
            col1, col2 = st.columns([3,2])

            with col1:
                wordcloud = get_wordcloud(nearby_places[i]['global_index'], tokens)
                st.image(wordcloud.to_array(), width=600)

            with col2:
                random_indices = random.sample(range(0, len(nearby_places[i]['review_text'])), 5)

                for idx, random_index in enumerate(random_indices):
                    expander = st.expander(f"Review {idx + 1}")
                    with expander:
                        st.write(nearby_places[i]['review_text'][random_index])

            photos = ast.literal_eval(nearby_places[i]['review_photos'])[:20]
            if isinstance(photos, list):
                st.image(photos)


else:
    st.write("Simply input the keywords and locale you wish to search for!")
    if location:
        data = [{"lat": float(lat), "lon": float(lon)}]

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=data,
            get_position=["lon", "lat"],
            get_radius=10,
            get_color='[255, 0, 0]',
            pickable=True,
        )

        view_state = pdk.ViewState(
            latitude=float(lat),
            longitude=float(lon),
            zoom=16,
            pitch=0,
        )

        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/streets-v11',
            initial_view_state=view_state,
            layers=[layer],
        ))
    else:
        st.error("Failed to get geolocation. Please ensure your browser allows location access and try again.")
