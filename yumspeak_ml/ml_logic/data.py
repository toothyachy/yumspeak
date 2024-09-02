import pandas as pd
import numpy as np
import re
import ast
import geocoder

from yumspeak_ml.params import *

def find_coordinates(link):
    # get coordinates from link url
    match = re.search('!3d(-?\d+(?:\.\d+)?)!4d(-?\d+(?:\.\d+))', link)
    coordinates = [float(match.group(1)), float(match.group(2))]
    return coordinates

def add_lat_lng(df):
    # get coordinates, + latitude, longtitude
    df['coordinates'] = df['link'].apply(lambda x: find_coordinates(x))
    df['latitude'] = df['coordinates'].apply(lambda x: x[0])
    df['longtitude'] = df['coordinates'].apply(lambda x: x[1])
    return df

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


# clean dataset
def clean_restaurant_data(df: pd.DataFrame) -> pd.DataFrame:
    #drop irrelevant columns, rows with null values, df duplicates with subset 'place_id', 'name', 'reviews', 'address']
    df = df.drop(columns=['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16'])
    df = df.dropna()

    # remove irrelevant place based on main_categories
    cats_to_remove = [cat.lower() for cat in CATS_TO_REMOVE]
    mask = df['main_category'].str.lower().isin(cats_to_remove)
    df = df[~mask]

    # remove irrelevant place based by name and place_id
    df = df[~df['name'].isin(NAMES_TO_DROP)]
    df = df[~df['place_id'].isin(PLACE_ID_TO_DROP)]

    # main_category 'Restaurant', retrieve category from categories if there are more than 1 category.
    sec_cat_mask = (df['main_category'] == 'Restaurant') & (df['categories'].apply(lambda x: x != ['Restaurant']))
    df.loc[sec_cat_mask, 'main_category'] = df['categories'].apply(lambda x: x[1] if len(x) > 1 else x[0])

    # recategorize remaining restaurants and remove ' restaurant' from the data
    restaurant_cat_mask = df['main_category'] == 'Restaurant'
    df.loc[restaurant_cat_mask, 'main_category'] = df['name'].apply(lambda x: RESTAURANT_RECATEGORIZATION.get(x, 'Restaurant'))
    df['main_category'] = df['main_category'].apply(lambda x: x.replace(' restaurant', ''))
    df = df.drop_duplicates(subset=['place_id', 'name', 'reviews', 'address'])

    # get lat lng from link (+coordinates, latitude, longitude)
    df = add_lat_lng(df)

    # filter out places with coordinates outside of SG
    df = df[df['latitude'].between(left=1.129, right=1.493)]
    df = df[df['longtitude'].between(left=103.557, right=104.131)]
    df = df.reset_index(drop=True)

    # get postal code with with mapbox reverse geocoding, extract district_code and region
    df = df.apply(get_postal_code, axis=1)
    df['district_code'] = df['postal_code'].map(POSTAL_TO_DISTRICT)
    df['region'] = df['district_code'].map(DISTRICT_TO_REGION)

    df = df.dropna()

    return df
