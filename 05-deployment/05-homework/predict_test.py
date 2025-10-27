import pickle
from flask import Flask
from flask import request
from flask import jsonify


model_file = 'pipeline_v1.bin'

with open(model_file,'rb') as f_in:
    dv,model = pickle.load(f_in)


app = Flask('converted')

@app.route('/predict_test', methods=['POST'])


def predict():

    record = request.get_json()

    X = dv.transform([record])
    y_pred = model.predict_proba(X)[0,1]
    converted = y_pred >= 0.5

    result = {
        "convertion_probability" : float(round(y_pred,3)),
        "converted" : bool(converted)
    }

    return jsonify(result)


if __name__== "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)