from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import joblib
import json
from PIL import Image
import os
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize models as None
cnn_model = None
knn_model = None
dataset = None

def create_knn_model(df):
    """Create and train KNN model"""
    try:
        features = df[['Karbohidrat (g)', 'Protein (g)', 'Lemak (g)']].values
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(features)
        logger.info("KNN model created successfully")
        return knn
    except Exception as e:
        logger.error(f"Error creating KNN model: {str(e)}")
        raise

def load_models():
    """Function to load all required models"""
    global cnn_model, knn_model, dataset
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    try:
        model_path = os.path.join(base_path, 'model', 'v1.h5')
        logger.info(f"Loading CNN model from: {model_path}")
        cnn_model = tf.keras.models.load_model(model_path, compile=False)
        cnn_model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        logger.info("CNN model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading CNN model: {str(e)}")
        raise

    try:
        dataset_path = os.path.join(base_path, 'dataset.json')
        logger.info(f"Loading dataset from: {dataset_path}")
        with open(dataset_path) as f:
            dataset = json.load(f)
        logger.info("Dataset loaded successfully")
        
        df = pd.DataFrame(dataset)
        
        knn_model = create_knn_model(df)
        logger.info("KNN model created successfully")
        
    except Exception as e:
        logger.error(f"Error loading dataset or creating KNN model: {str(e)}")
        raise

# Try to load models when app starts
try:
    logger.info("Starting model loading...")
    load_models()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    pass

@app.route('/status', methods=['GET'])
def check_status():
    """Check if models are loaded properly"""
    return jsonify({
        'cnn_model_loaded': cnn_model is not None,
        'knn_model_loaded': knn_model is not None,
        'dataset_loaded': dataset is not None
    })

def predict_food(image_path):
    """Predict food from image and get nutrition info"""
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = cnn_model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))
        
        food_name = dataset[predicted_class]["Nama Makanan/Minuman"]
        nutrition = {
            "kalori": dataset[predicted_class]["Kalori (kcal)"],
            "karbohidrat": dataset[predicted_class]["Karbohidrat (g)"],
            "protein": dataset[predicted_class]["Protein (g)"],
            "lemak": dataset[predicted_class]["Lemak (g)"]
        }
        return food_name, nutrition, confidence
    except Exception as e:
        logger.error(f"Error in predict_food: {str(e)}")
        raise

def get_food_recommendations(food_features):
    """Get food recommendations using KNN model"""
    try:
        # Convert food_features to numpy array and reshape if needed
        food_features = np.array(food_features).reshape(1, -1)
        
        # Get nearest neighbors
        distances, indices = knn_model.kneighbors(food_features)
        
        recommendations = []
        for i, idx in enumerate(indices[0]):
            food = dataset[int(idx)]
            recommendations.append({
                "nama": food["Nama Makanan/Minuman"],
                "nutrition": {
                    "kalori": float(food["Kalori (kcal)"]),  # Convert to float
                    "karbohidrat": float(food["Karbohidrat (g)"]),
                    "protein": float(food["Protein (g)"]),
                    "lemak": float(food["Lemak (g)"])
                },
                "similarity_score": float(1 / (1 + distances[0][i]))
            })
        return recommendations
    except Exception as e:
        logger.error(f"Error in get_food_recommendations: {str(e)}")
        raise

def hitung_bmr_tdee(berat_badan, tinggi_badan, umur, jenis_kelamin, tingkat_aktivitas):
    """Calculate BMR and TDEE"""
    if jenis_kelamin.lower() == 'pria':
        bmr = 10 * berat_badan + 6.25 * tinggi_badan - 5 * umur + 5
    else:  # wanita
        bmr = 10 * berat_badan + 6.25 * tinggi_badan - 5 * umur - 161

    activity_multipliers = {
        "ringan": 1.375,
        "sedang": 1.55,
        "berat": 1.725
    }
    
    if tingkat_aktivitas.lower() not in activity_multipliers:
        raise ValueError("Tingkat aktivitas tidak valid. Pilih 'ringan', 'sedang', atau 'berat'.")
        
    tdee = bmr * activity_multipliers[tingkat_aktivitas.lower()]
    return bmr, tdee

def hitung_kebutuhan_makronutrien(tdee):
    """Calculate macronutrient needs"""
    kalori_karbohidrat = tdee * 0.55
    kalori_protein = tdee * 0.20
    kalori_lemak = tdee * 0.25

    gram_karbohidrat = kalori_karbohidrat / 4
    gram_protein = kalori_protein / 4
    gram_lemak = kalori_lemak / 9

    return gram_karbohidrat, gram_protein, gram_lemak

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for food prediction from image"""
    if cnn_model is None or dataset is None:
        return jsonify({'error': 'Models not loaded properly'}), 503
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    try:
        file = request.files['file']
        # Create temp directory if it doesn't exist
        os.makedirs('temp', exist_ok=True)
        
        file_path = os.path.join('temp', file.filename)
        file.save(file_path)
        
        food_name, nutrition, confidence = predict_food(file_path)
        
        # Clean up temporary file
        os.remove(file_path)
        
        # Get food features for recommendation
        food_features = [
            nutrition["karbohidrat"],
            nutrition["protein"], 
            nutrition["lemak"]
        ]
        recommendations = get_food_recommendations(food_features)
        
        return jsonify({
            'food_name': food_name,
            'confidence': confidence,
            'nutrition': nutrition,
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/calculate', methods=['POST'])
def calculate():
    """Endpoint for calculating nutrition needs"""
    try:
        data = request.json
        required_fields = ['weight', 'height', 'age', 'gender', 'activity_level']
        
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        bmr, tdee = hitung_bmr_tdee(
            data['weight'],
            data['height'],
            data['age'],
            data['gender'],
            data['activity_level']
        )
        
        karbohidrat, protein, lemak = hitung_kebutuhan_makronutrien(tdee)
        
        return jsonify({
            'bmr': float(bmr),
            'tdee': float(tdee),
            'kebutuhan_harian': {
                'karbohidrat': float(karbohidrat),
                'protein': float(protein),
                'lemak': float(lemak)
            }
        })
        
    except Exception as e:
        logger.error(f"Error in calculate endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Endpoint for food recommendations based on nutrition values"""
    if knn_model is None or dataset is None:
        logger.error("Models not loaded properly")
        return jsonify({'error': 'Models not loaded properly'}), 503
        
    try:
        data = request.json
        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        required_fields = ['karbohidrat', 'protein', 'lemak']
        
        if not all(field in data for field in required_fields):
            missing_fields = [field for field in required_fields if field not in data]
            logger.error(f"Missing fields: {missing_fields}")
            return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400
        
        # Validate numeric values and convert to float
        try:
            food_features = [
                float(data['karbohidrat']),
                float(data['protein']),
                float(data['lemak'])
            ]
        except ValueError as e:
            logger.error(f"Invalid numeric values: {str(e)}")
            return jsonify({'error': 'All nutritional values must be numbers'}), 400
            
        # Validate non-negative values
        if any(x < 0 for x in food_features):
            logger.error("Negative nutritional values provided")
            return jsonify({'error': 'Nutritional values cannot be negative'}), 400
        
        recommendations = get_food_recommendations(food_features)
        
        if not recommendations:
            logger.warning("No recommendations found")
            return jsonify({'message': 'No similar foods found', 'recommendations': []}), 200
        
        return jsonify({
            'input_values': {
                'karbohidrat': food_features[0],
                'protein': food_features[1],
                'lemak': food_features[2]
            },
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error occurred'}), 500

@app.route('/recommend-by-name', methods=['POST'])
def recommend_by_name():
    """Endpoint for food recommendations based on food name"""
    if knn_model is None or dataset is None:
        return jsonify({'error': 'Models not loaded properly'}), 503
        
    try:
        data = request.json
        if 'food_name' not in data:
            return jsonify({'error': 'Missing food_name field'}), 400
            
        # Find the food in dataset
        food_name = data['food_name'].lower().strip()  # Add strip() to remove whitespace
        food_data = None
        
        for i, food in enumerate(dataset):
            if food["Nama Makanan/Minuman"].lower().strip() == food_name:
                food_data = food
                break
                
        if food_data is None:
            return jsonify({'error': 'Food not found in database'}), 404
            
        # Get food features and convert to float
        food_features = [
            float(food_data["Karbohidrat (g)"]),
            float(food_data["Protein (g)"]),
            float(food_data["Lemak (g)"])
        ]
        
        recommendations = get_food_recommendations(food_features)
        
        return jsonify({
            'input_food': {
                'nama': food_data["Nama Makanan/Minuman"],
                'nutrition': {
                    'kalori': food_data["Kalori (kcal)"],
                    'karbohidrat': food_data["Karbohidrat (g)"],
                    'protein': food_data["Protein (g)"],
                    'lemak': food_data["Lemak (g)"]
                }
            },
            'recommendations': recommendations
        })
        
    except Exception as e:
        logger.error(f"Error in recommend-by-name endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


