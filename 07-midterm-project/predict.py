import pickle
import xgboost as xgb
from flask import Flask, request, jsonify

model_file = "model_eta=0.1_max_depth=10_min_child_weight=1.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("delivery_time")


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    features = list(dv.get_feature_names_out())
    X = dv.transform([customer])
    
    dx = xgb.DMatrix(X, feature_names=features)
    
    y_pred = model.predict(dx)

    result = {
        "delivery_time_prediction_in_min": round(float(y_pred), 3)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
