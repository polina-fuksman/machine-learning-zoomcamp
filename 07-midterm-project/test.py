import requests



url = 'http://localhost:9696/predict'



customer = {
    "delivery_person_age": 50.0,
    "delivery_person_ratings": 4.7,
    "weather_conditions": "fog",
    "road_traffic_density": "low",
    "vehicle_condition": "unknown",
    "type_of_order": "snack",
    "type_of_vehicle": "motorcycle",
    "multiple_deliveries": 1.0,
    "festival": "no",
    "city": "metropolitian",
    "distance_km": 100.0,
    "order_hour": 8,
    "order_day_of_week": 6,
    "preparation_time_mins": 150.0
 }



response = requests.post(url, json=customer).json()

print(response)



if response['delivery_time_prediction_in_min'] > 25:
    print(f"Too far, will be at the address only in {round(response['delivery_time_prediction_in_min'],0) + 1} minutes")
else:
    print(f"Will be at the address in {round(response['delivery_time_prediction_in_min'],0) + 1} minutes")

