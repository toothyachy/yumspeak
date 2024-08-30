import pandas as pd
import numpy as np
import re
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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    #drop duplicates with subset 'place_id', 'name', 'reviews', 'address']
    df = df.drop_duplicates(subset=['place_id', 'name', 'reviews', 'address'])

    # remove irrelevant place based on main_categories
    cats_to_remove = [cat.lower() for cat in CAT_TO_REMOVE]
    mask = df['main_category'].str.lower().isin(cats_to_remove)
    df = df[~mask]

    # fill na in main_category and categories(as ['unknown'])
    df['main_categories'] = df['main_categories'].fillna('unknown', inplace=True)
    df['categories'] = df['categories'].fillna("['unknown']").apply(eval)

    # get lat lng from link (+coordinates, latitude, longitude)
    df = add_lat_lng(df)

    # filter out places with coordinates outside of SG
    df = df[df['latitude'].between(left=1.129, right=1.493)]
    df = df[df['longtitude'].between(left=103.557, right=104.131)]

    df = df.reset_index(drop=True)

    return df



##########
def get_district_code(row):
    if isinstance(row['address'], str):
        try:
            match = re.search(r'\b\d{6}\b', row['address'])
            postal_code = match.group(0)
            row['district_code'] = postal_code[:2]
            return row
        except:
            g =geocoder.mapbox(row['coordinates'], method='reverse', key=MAP_API)
            row['address'] = g.json['postal']
            row['district_code'] = g.json['postal'][:2]
            return row
    else:
        try:
            g =geocoder.mapbox(row['coordinates'], method='reverse', key=MAP_API)
            row['address'] = g.json['postal']
            row['district_code'] = g.json['postal'][:2]
            return row
        except:
            return(row['address'])
