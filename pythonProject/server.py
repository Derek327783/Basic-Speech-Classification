# Client (Sends audio files via POST request) --> Server (Server takes in and uses model) --> Sends predictions
import random
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service
import os


app = Flask(__name__)

@app.route("/predict", methods = ["POST"])
def predict():
    # get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0,1000))
    audio_file.save(file_name)

    # invoke keyword spotting service
    service = Keyword_Spotting_Service()

    # make a prediction
    prediction = service.predict(file_name)

    # remove audio file
    os.remove(file_name)
    # send back predicted keyword in json format
    data = {"keyword":prediction}
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=False)
