from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Models

model = pickle.load(
    open("models/crop_model.pkl", "rb")
)

encoder = pickle.load(
    open("models/label_encoder.pkl", "rb")
)

kmeans = pickle.load(
    open("models/kmeans_model.pkl", "rb")
)

try:
    with open("models/model_name.txt", "r") as f:
        model_name = f.read().strip()
except:
    model_name = "Random Forest"

# Crop Information

crop_info = {

    "rice":{
        "season":"Kharif",
        "water":"High",
        "soil":"Clay Loam",
        "climate":"Warm and Humid"
    },

    "maize":{
        "season":"Kharif/Rabi",
        "water":"Medium",
        "soil":"Loamy Soil",
        "climate":"Moderate"
    },

    "chickpea":{
        "season":"Rabi",
        "water":"Low",
        "soil":"Sandy Loam",
        "climate":"Cool and Dry"
    },

    "kidneybeans":{
        "season":"Kharif",
        "water":"Medium",
        "soil":"Well Drained Soil",
        "climate":"Warm"
    },

    "pigeonpeas":{
        "season":"Kharif",
        "water":"Medium",
        "soil":"Loamy Soil",
        "climate":"Semi-Arid"
    },

    "mothbeans":{
        "season":"Kharif",
        "water":"Low",
        "soil":"Sandy Soil",
        "climate":"Dry"
    },

    "mungbean":{
        "season":"Kharif",
        "water":"Medium",
        "soil":"Loamy Soil",
        "climate":"Warm"
    },

    "blackgram":{
        "season":"Kharif",
        "water":"Medium",
        "soil":"Clay Loam",
        "climate":"Warm and Humid"
    },

    "lentil":{
        "season":"Rabi",
        "water":"Low",
        "soil":"Loamy Soil",
        "climate":"Cool"
    },

    "pomegranate":{
        "season":"Perennial",
        "water":"Medium",
        "soil":"Well Drained Soil",
        "climate":"Tropical"
    },

    "banana":{
        "season":"Year Round",
        "water":"Very High",
        "soil":"Rich Loamy Soil",
        "climate":"Hot and Humid"
    },

    "mango":{
        "season":"Summer",
        "water":"Medium",
        "soil":"Deep Loamy Soil",
        "climate":"Tropical"
    },

    "grapes":{
        "season":"Winter",
        "water":"Medium",
        "soil":"Well Drained Soil",
        "climate":"Dry"
    },

    "watermelon":{
        "season":"Summer",
        "water":"Medium",
        "soil":"Sandy Loam",
        "climate":"Warm"
    },

    "muskmelon":{
        "season":"Summer",
        "water":"Medium",
        "soil":"Sandy Loam",
        "climate":"Warm"
    },

    "apple":{
        "season":"Winter",
        "water":"Medium",
        "soil":"Loamy Soil",
        "climate":"Cool"
    },

    "orange":{
        "season":"Winter",
        "water":"Medium",
        "soil":"Well Drained Soil",
        "climate":"Subtropical"
    },

    "papaya":{
        "season":"Year Round",
        "water":"Medium",
        "soil":"Loamy Soil",
        "climate":"Tropical"
    },

    "coconut":{
        "season":"Year Round",
        "water":"High",
        "soil":"Sandy Coastal Soil",
        "climate":"Humid Tropical"
    },

    "cotton":{
        "season":"Kharif",
        "water":"Medium",
        "soil":"Black Soil",
        "climate":"Warm"
    },

    "jute":{
        "season":"Kharif",
        "water":"High",
        "soil":"Alluvial Soil",
        "climate":"Warm and Humid"
    },

    "coffee":{
        "season":"Perennial",
        "water":"Medium",
        "soil":"Rich Organic Soil",
        "climate":"Cool and Wet"
    }
}

seasonal_crops = {

    "Kharif":[
        "Rice",
        "Maize",
        "Cotton",
        "Jute",
        "Blackgram",
        "Mungbean",
        "Pigeonpeas"
    ],

    "Rabi":[
        "Chickpea",
        "Lentil"
    ],

    "Summer":[
        "Watermelon",
        "Muskmelon",
        "Mango"
    ]
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:

        n = float(request.form["N"])
        p = float(request.form["P"])
        k = float(request.form["K"])
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        values = np.array([
            [n, p, k, temp, humidity, ph, rainfall]
        ])

        prediction = model.predict(values)

        crop = encoder.inverse_transform(
            prediction
        )[0]

        cluster = int(
            kmeans.predict(values)[0]
        )

        try:

            probabilities = model.predict_proba(values)

            score = round(
                np.max(probabilities) * 100,
                2
            )

        except:

            score = 95.0

        info = crop_info.get(
            crop.lower(),
            {
                "season":"Not Available",
                "water":"Not Available",
                "soil":"Not Available",
                "climate":"Not Available"
            }
        )

        inputs = {

            "Nitrogen": n,
            "Phosphorous": p,
            "Potassium": k,
            "Temperature": temp,
            "Humidity": humidity,
            "pH": ph,
            "Rainfall": rainfall
        }

        return render_template(
            "result.html",
            crop=crop,
            score=score,
            cluster=cluster,
            info=info,
            inputs=inputs,
            model_name=model_name,
            seasonal_crops=seasonal_crops
        )

    except Exception as e:

        return f"Error : {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)