from flask import Flask, request, jsonify

from response import parse_model_response
from cpm_classifier import get_cpm_classifier

app = Flask(__name__)

cpm_classifier = get_cpm_classifier()

@app.route('/cpm-classifier/predict', methods=['POST'])
def classifier():
    body = request.get_json()

    if (body is None) or (type(body) is not list):
        return "Please include a list of objects in JSON Body", 400

    model_inputs, errors = cpm_classifier.parse_samples(body)
    
    predicts = cpm_classifier.predict(model_inputs)

    response = parse_model_response(predicts, errors)

    return jsonify(response), 200

app.run(debug=True)