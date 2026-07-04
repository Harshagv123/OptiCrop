# 🌱 OptiCrop

    Smart Agricultural Production Optimization Engine

OptiCrop is an AI-powered crop recommendation system that suggests the most suitable crop based on soil nutrients and environmental conditions. It uses Machine Learning models and a Flask web application to provide fast and accurate predictions.



 ✨ Features

- 🌾 Crop Recommendation
- 🌱 Soil Nutrient Analysis
- 🌦 Weather Parameter Analysis
- 🤖 Machine Learning Prediction
- 📊 Data Visualization
- 💻 Flask Web Application



 🛠 Tech Stack

- Python
- Flask
- HTML, CSS, JavaScript
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib



 📂 Project Structure   

        OptiCrop/
OptiCrop/
│
├── app.py                      # Main Flask application
├── train_model.py              # Model training script
├── requirements.txt            # Required Python packages
├── README.md                   # Project documentation
│
├── data/
│   └── Crop_recommendation.csv # Agricultural dataset
│
├── models/
│   ├── crop_model.pkl          # Trained crop prediction model
│   ├── kmeans_model.pkl        # K-Means clustering model
│   ├── label_encoder.pkl       # Label encoder
│   └── model_name.txt          # Model information
│
├── notebooks/
│   └── OptiCrop_Analysis.ipynb # Data analysis & EDA notebook
│
├── templates/
│   ├── index.html              # User input page
│   └── result.html             # Prediction result page
│
├── static/
     └── images/
         ├── correlation_heatmap.png
         ├── crop_distribution.png
         └── temperature_rainfall.png

 📊 Dataset

- 2200 Records
- 22 Crop Classes


- Input Features:

  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Temperature
  - Humidity
  - pH
  - Rainfall



  Run the Project

______bash________
pip install -r requirements.txt
python app.py


Open:

http://127.0.0.1:5000


 👥 Team

- Harsha
- Swetha
- Sneha
- Pranavi
- Jagan
