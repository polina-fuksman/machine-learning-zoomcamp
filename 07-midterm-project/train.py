# Libraries

import pickle

import pandas as pd
import numpy as np

from haversine import haversine, Unit

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

import xgboost as xgb

# XGBoost Params

eta=0.1
max_depth=10
min_child_weight=1

output_file = f'model_eta={eta}_max_depth={max_depth}_min_child_weight={min_child_weight}.bin'


# Data

# source https://huggingface.co/datasets/aneesarom/Food-Delivery

#!wget -O food_delivery_test.parquet "https://huggingface.co/datasets/aneesarom/Food-Delivery/resolve/main/data/test-00000-of-00001.parquet?download=true"
#!wget -O food_delivery_train.parquet "https://huggingface.co/datasets/aneesarom/Food-Delivery/resolve/main/data/train-00000-of-00001.parquet?download=true"


df_test_raw = pd.read_parquet('food_delivery_test.parquet')
df_train_raw = pd.read_parquet('food_delivery_train.parquet')

df = pd.concat([df_train_raw, df_test_raw], ignore_index=True)


# Data cleaning


df.columns = df.columns.str.lower().str.replace(' ','_').str.replace('(','').str.replace(')','').str.replace('time_orderd', 'time_ordered')


columns = list(df.columns[df.dtypes == 'object'])

for col in columns:
    df[col] = df[col].str.lower().str.replace(' ','_')



# Null values

df = df[df['multiple_deliveries'].notna()]

df['time_ordered'] = df['time_ordered'].fillna('unknown')
df['weather_conditions'] = df['weather_conditions'].fillna('unknown')
df['road_traffic_density'] = df['road_traffic_density'].fillna('unknown')
df['festival'] = df['festival'].fillna('unknown')
df['city'] = df['city'].fillna('unknown')


vehicle_conditions = {
    1:'ok', 
    2:'small_damage',
    3:'damage',
    0:'unknown'
}

df.vehicle_condition = df.vehicle_condition.map(vehicle_conditions)


# Features that are important for delivery time prediction

def calculate_distance(row):
    restaurant_coords = (row['restaurant_latitude'], row['restaurant_longitude'])
    delivery_coords = (row['delivery_location_latitude'], row['delivery_location_longitude'])
    return haversine(restaurant_coords, delivery_coords, unit=Unit.KILOMETERS)

df['distance_km'] = df.apply(calculate_distance, axis=1)



df['order_timestamp'] = pd.to_datetime(
    df['order_date'] + ' ' + df['time_ordered'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

df['picked_timestamp'] = pd.to_datetime(
    df['order_date'] + ' ' + df['time_order_picked'],
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

mask_24h = df['picked_timestamp'].isna() & df['time_order_picked'].str.startswith('24:', na=False)


if mask_24h.any():

    next_day_date = pd.to_datetime(df.loc[mask_24h, 'order_date'], format='%d-%m-%Y') + pd.Timedelta(days=1)


    corrected_time = df.loc[mask_24h, 'time_order_picked'].str.replace('24:', '00:', 1)


    df.loc[mask_24h, 'picked_timestamp'] = pd.to_datetime(
        next_day_date.dt.strftime('%Y-%m-%d') + ' ' + corrected_time
    )


df['order_hour'] = df['order_timestamp'].dt.hour
df['order_day_of_week'] = df['order_timestamp'].dt.dayofweek
df['preparation_time_mins'] = (df['picked_timestamp'] - df['order_timestamp']).dt.total_seconds() / 60


df = df[df['preparation_time_mins'].notna()]


df_prepared = df[['delivery_person_age', 'delivery_person_ratings', 'weather_conditions', 'road_traffic_density', 'vehicle_condition', 
    'type_of_order', 'type_of_vehicle', 'multiple_deliveries', 'festival', 'city', 'time_taken_min',
                 'distance_km', 'order_hour', 'order_day_of_week', 'preparation_time_mins']]


# Data split and fill numerical null values with medians


df_full_train, df_test = train_test_split(df_prepared, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


y_full_train = df_full_train.time_taken_min.values
y_train = df_train.time_taken_min.values
y_val = df_val.time_taken_min.values
y_test = df_test.time_taken_min.values


del df_full_train['time_taken_min']
del df_train['time_taken_min']
del df_val['time_taken_min']
del df_test['time_taken_min']



median_delivery_person_age = df_full_train['delivery_person_age'].median()
median_delivery_person_ratings = df_full_train['delivery_person_ratings'].median()

df_full_train['delivery_person_age'] = df_full_train['delivery_person_age'].fillna(median_delivery_person_age)
df_full_train['delivery_person_ratings'] = df_full_train['delivery_person_ratings'].fillna(median_delivery_person_ratings)



median_delivery_person_age = df_train['delivery_person_age'].median()
median_delivery_person_ratings = df_train['delivery_person_ratings'].median()

df_train['delivery_person_age'] = df_train['delivery_person_age'].fillna(median_delivery_person_age)
df_train['delivery_person_ratings'] = df_train['delivery_person_ratings'].fillna(median_delivery_person_ratings)



median_delivery_person_age = df_val['delivery_person_age'].median()
median_delivery_person_ratings = df_val['delivery_person_ratings'].median()

df_val['delivery_person_age'] = df_val['delivery_person_age'].fillna(median_delivery_person_age)
df_val['delivery_person_ratings'] = df_val['delivery_person_ratings'].fillna(median_delivery_person_ratings)



median_delivery_person_age = df_test['delivery_person_age'].median()
median_delivery_person_ratings = df_test['delivery_person_ratings'].median()

df_test['delivery_person_age'] = df_test['delivery_person_age'].fillna(median_delivery_person_age)
df_test['delivery_person_ratings'] = df_test['delivery_person_ratings'].fillna(median_delivery_person_ratings)



# DictVectorizer

dv = DictVectorizer(sparse=True)

full_train_dicts = df_full_train.to_dict(orient='records')
test_dicts = df_test.to_dict(orient='records')

X_full_train = dv.fit_transform(full_train_dicts)
X_test = dv.transform(test_dicts)

features = list(dv.get_feature_names_out())

dfull_train = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features)
dtest = xgb.DMatrix(X_test, feature_names=features)


# training the model

xgb_params = {
    'eta': eta,
    'max_depth': max_depth,
    'min_child_weight': min_child_weight,

    'objective': 'reg:squarederror',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfull_train, num_boost_round=80)

print(f"Training the model with {xgb_params}")


# testing the model

def rmse(y, y_pred):
    error = y - y_pred
    se = error ** 2
    mse = se.mean()
    return np.sqrt(mse)

y_pred = model.predict(dtest)
score = rmse(y_test, y_pred)

print(f"The RMSE of the model is {score}")

# Saving model


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print (f"Saved the model in {output_file}")