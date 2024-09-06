import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def KNN_norm_cord_search(Model, coordinate, search_area = 3):
    features = Model[['latitude', 'longtitude']]
    target = Model['main_category']

    KNN = KNeighborsClassifier(metric='haversine')
    KNN.fit(features.values, target)
    neighbors = KNN.kneighbors(coordinate, n_neighbors=search_area, return_distance=False)
    selected_rows = Model.iloc[neighbors[0]][['name', 'latitude', 'longtitude']]
    data_dict = selected_rows.to_dict(orient='records')
    return data_dict
