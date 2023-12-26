import os
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def predict():

    if request.method == 'POST':
        img = request.files['imagefile']
        img_path = "./image/" + img.filename
        img.save(img_path)

        img = image.load_img(img_path, target_size=(120, 120), color_mode="grayscale")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=-1)
        x /= 255.0

        json_file = open("./models/affect_model.json", "r")

        loaded_model_json = json_file.read()

        json_file.close()

        loaded_model_v = model_from_json(loaded_model_json)
        loaded_model_a = model_from_json(loaded_model_json)

        loaded_model_v.load_weights("./models/valence-weights-improvement-137-0.23.h5")
        loaded_model_a.load_weights("./models/arousal-weights-improvement-95-0.17.h5")

        loaded_model_v.compile(loss='mean_squared_error', optimizer='sgd')
        loaded_model_a.compile(loss='mean_squared_error', optimizer='sgd')

        prediction_v = loaded_model_v.predict(x)
        prediction_a = loaded_model_a.predict(x)

        # print(f"valence value {prediction_v}")
        # print(f"arousal value {prediction_a}")
    
        return render_template("index.html", prediction_v=prediction_v, prediction_a=prediction_a)
    return render_template("index.html")

if __name__ =='__main__':
    app.run(debug=True, port=8090)