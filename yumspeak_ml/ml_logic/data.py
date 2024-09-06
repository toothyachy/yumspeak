import pandas as pd
import numpy as np
import re
import ast
import geocoder
from datetime import datetime

from yumspeak_ml.params import *

def find_coordinates(link):
    # get coordinates from link url
    match = re.search('!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+))', link)
    coordinates = [float(match.group(1)), float(match.group(2))]
    return coordinates

def add_lat_lng(restaurants_df):
    # get coordinates, + latitude, longitude
    restaurants_df['coordinates'] = restaurants_df['link'].apply(lambda x: find_coordinates(x))
    restaurants_df['latitude'] = restaurants_df['coordinates'].apply(lambda x: x[0])
    restaurants_df['longitude'] = restaurants_df['coordinates'].apply(lambda x: x[1])
    return restaurants_df

def get_postal_code(row):
    # get postal code with with mapbox reverse geocoding
    if isinstance(row['address'], str):
        try:
            match = re.search(r'\b\d{6}\b', row['address'])
            postal_code = match.group(0)
            row['full_postal_code'] = postal_code
            row['postal_code'] = postal_code[:2]
            return row
        except:
            g =geocoder.mapbox(row['coordinates'], method='reverse', key=MAP_API)
            row['address'] = f"Singapore {g.json['postal']}"
            row['full_postal_code'] = g.json['postal']
            row['postal_code'] = g.json['postal'][:2]
            return row
    else:
        try:
            g =geocoder.mapbox(row['coordinates'], method='reverse', key=MAP_API)
            row['address'] = f"Singapore {g.json['postal']}"
            row['full_postal_code'] = g.json['postal']
            row['postal_code'] = g.json['postal'][:2]
            return row
        except:
            print(f"Error: {row['address']}")
            return row['address']


# clean and merge dataset after scraping
def clean_data(restaurants_df:pd.DataFrame, reviews_df:pd.DataFrame) -> pd.DataFrame:
    # restaurant df
    # drop irrelevant columns, rows with null values, restaurants_df duplicates with subset 'place_id', 'name', 'reviews', 'address']
    restaurants_df = restaurants_df.drop(columns=['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'])
    restaurants_df = restaurants_df.drop_duplicates(subset=['place_id', 'name', 'address'], keep='last')
    restaurants_df = restaurants_df.dropna()

    # remove irrelevant place based by name and place_id
    restaurants_df = restaurants_df[~restaurants_df['name'].isin(NAMES_TO_DROP)]
    restaurants_df = restaurants_df[~restaurants_df['place_id'].isin(PLACE_ID_TO_DROP)]

    # filter out main_rating < 3.7 review counts >=75 <=2000
    restaurants_df = restaurants_df[(restaurants_df['main_rating'] >= 3.7) & (restaurants_df['reviews'] >= 75) & (restaurants_df['reviews'] <= 2000)]

    # main_category 'Restaurant', retrieve category from categories if there are more than 1 category.
    restaurants_df['categories'] = restaurants_df['categories'].apply(lambda x: ast.literal_eval(x))
    sec_cat_mask = (restaurants_df['main_category'] == 'Restaurant') & (restaurants_df['categories'].apply(lambda x: x != ['Restaurant']))
    restaurants_df.loc[sec_cat_mask, 'main_category'] = restaurants_df['categories'].apply(lambda x: x[1] if len(x) > 1 else x[0])

    # recategorize remaining restaurants
    restaurant_cat_mask = restaurants_df['main_category'] == 'Restaurant'
    restaurants_df.loc[restaurant_cat_mask, 'main_category'] = restaurants_df['name'].apply(lambda x: RESTAURANT_RECATEGORIZATION.get(x, 'Restaurant'))

    # remove irrelevant place based on main_categories
    cats_to_remove = [cat.lower() for cat in CATS_TO_REMOVE]
    mask = restaurants_df['main_category'].str.lower().isin(cats_to_remove)
    restaurants_df = restaurants_df[~mask]

    # remove ' restaurant' from the data
    restaurants_df['main_category'] = restaurants_df['main_category'].apply(lambda x: x.replace(' restaurant', ''))

    # map main_category to cuisine, reducing unique values
    restaurants_df['cuisine'] = restaurants_df['main_category'].apply(lambda x: CATEGORY_TO_CUISINE[x])

    # get lat lng from link (+coordinates, latitude, longitude)
    restaurants_df = add_lat_lng(restaurants_df)

    # filter out places with coordinates outside of SG
    restaurants_df = restaurants_df[restaurants_df['latitude'].between(left=1.129, right=1.493)]
    restaurants_df = restaurants_df[restaurants_df['longitude'].between(left=103.557, right=104.131)]
    restaurants_df = restaurants_df.reset_index(drop=True)

    # get postal code with with mapbox reverse geocoding, extract district_code and region
    restaurants_df = restaurants_df.apply(get_postal_code, axis=1)
    restaurants_df['district_code'] = restaurants_df['postal_code'].map(POSTAL_TO_DISTRICT)
    restaurants_df['region'] = restaurants_df['district_code'].map(DISTRICT_TO_REGION)
    restaurants_df = restaurants_df[~(restaurants_df['full_postal_code'].str.len() < 6)]

    # drop all places not in SG
    restaurants_df = restaurants_df.dropna()

    # drop the coordinates column
    restaurants_df = restaurants_df.drop(columns=['coordinates', 'categories'])

    # reviews_df df
    # Drop duplicates
    reviews_df = reviews_df.drop_duplicates()

    # Drop unnecessary columns
    reviews_df = reviews_df.drop(columns=['review_translated_text', 'response_from_owner_translated_text', 'response_from_owner_ago', 'response_from_owner_date', 'published_at'])

    # Drop rows where review_text and published_at is null
    reviews_df = reviews_df.dropna(subset=['review_text', 'published_at_date'])

    # Change remaining NaN to False, empty string or zero
    reviews_df['is_local_guide'] = reviews_df['is_local_guide'].fillna(False)
    reviews_df[['total_number_of_photos_by_reviewer', 'total_number_of_reviews_by_reviewer']] = reviews_df[['total_number_of_photos_by_reviewer', 'total_number_of_reviews_by_reviewer']].fillna(0)
    reviews_df['response_from_owner_text'] = reviews_df['response_from_owner_text'].fillna("")

    # Establish the consideration set - only keep reviews_df with published_at date from 1 Jan 2023
    # Convert the 'published_at_date' column to datetime and simplify to just date
    reviews_df['published_at_date'] = pd.to_datetime(reviews_df['published_at_date'])
    reviews_df['published_at_date'] = reviews_df['published_at_date'].dt.date
    cut_off_date = datetime.strptime("2023-01-01", '%Y-%m-%d').date()
    reviews_df = reviews_df[reviews_df['published_at_date'] >= cut_off_date]

    # Drop unnecessary columns
    reviews_df = reviews_df.drop(columns=['name', 'review_id'])

    # Inner merge
    merged_df = restaurants_df.merge(reviews_df, how='inner', on='place_id')

    # Remove duplicates
    merged_df = merged_df.drop_duplicates()

    # drop unnecessary columns
    merged_df = merged_df.drop(columns=['reviews', 'main_category', 'full_postal_code', 'postal_code', 'district_code', 'region', 'rating', 'published_at_date', 'review_likes_count', 'response_from_owner_text', 'total_number_of_reviews_by_reviewer', 'total_number_of_photos_by_reviewer', 'is_local_guide'])

    return merged_df
