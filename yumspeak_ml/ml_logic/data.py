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

'''
    if isinstance(row['address'], str):
        try:
            match = re.search(r'\b\d{6}\b', row['address'])
            postal_code = match.group(0)
            row['postal_code'] = postal_code[:2]
            return row
        except:
            g =geocoder.mapbox(row['coordinates'], method='reverse', key=MAP_API)
            row['address'] = f"Singapore {g.json['postal']}"
            row['postal_code'] = g.json['postal'][:2]
            return row
    else:
        try:
            g =geocoder.mapbox(row['coordinates'], method='reverse', key=MAP_API)
            row['address'] = f"Singapore {g.json['postal']}"
            row['postal_code'] = g.json['postal'][:2]
            return row
        except:
            print(f"Error: {row['address']}")
            return row['address']
'''

def get_postal_code(row):
    # get postal code with with mapbox reverse geocoding
    try:
        g =geocoder.mapbox(row['coordinates'], method='reverse', key=MAP_API)
        row['full_postal_code'] = f"Singapore {g.json['postal']}"
        row['postal_code'] = g.json['postal'][:2]
        return row
    except:
        match = re.search(r'\b\d{6}\b', row['address'])
        row['full_postal_code'] = match.group(0)
        row['postal_code'] = row['full_postal_code'][:2]
        return row


# clean dataset
def clean_restaurant_data(df: pd.DataFrame) -> pd.DataFrame:
    #drop duplicates with subset 'place_id', 'name', 'reviews', 'address']
    df = df.drop_duplicates(subset=['place_id', 'name', 'reviews', 'address'])

    # remove irrelevant place based on main_categories
    cats_to_remove = [cat.lower() for cat in CATS_TO_REMOVE]
    mask = df['main_category'].str.lower().isin(cats_to_remove)
    df = df[~mask]

    # remove irrelevant place based by name
    #df = df[~df['name'].isin(NAME_TO_DROP)]
    df = df[~df['place_id'].isin(PLACE_ID_TO_DROP)]

    # fill na in main_category and categories(as ['unknown'])
    df['main_category'].fillna('unknown', inplace=True)
    df['categories'] = df['categories'].fillna("['unknown']").apply(ast.literal_eval)

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

    return df
