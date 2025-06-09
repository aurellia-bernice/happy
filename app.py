import joblib
import sqlite3
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load('rf_model_top10.pkl')
scaler = joblib.load('scaler_top10.pkl')

def init_db():
    conn = sqlite3.connect('BC.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            mean_radius REAL,
            mean_texture REAL,
            mean_perimeter REAL,
            mean_area REAL,
            mean_smoothness REAL,
            mean_compactness REAL,
            mean_concavity REAL,
            mean_concave_points REAL,
            mean_symmetry REAL,
            mean_fractal_dimension REAL,
            prediction TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_name = request.form['user_name']
    mean_radius = float(request.form['mean_radius'])
    mean_texture = float(request.form['mean_texture'])
    mean_perimeter = float(request.form['mean_perimeter'])
    mean_area = float(request.form['mean_area'])
    mean_smoothness = float(request.form['mean_smoothness'])
    mean_compactness = float(request.form['mean_compactness'])
    mean_concavity = float(request.form['mean_concavity'])
    mean_concave_points = float(request.form['mean_concave_points'])
    mean_symmetry = float(request.form['mean_symmetry'])
    mean_fractal_dimension = float(request.form['mean_fractal_dimension'])

    features = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                          mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension]])
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    result = "Malignant" if prediction[0] == 0 else "Benign"

    conn = sqlite3.connect('BC.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (
            user_name, mean_radius, mean_texture, mean_perimeter, mean_area,
            mean_smoothness, mean_compactness, mean_concavity,
            mean_concave_points, mean_symmetry, mean_fractal_dimension, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_name, mean_radius, mean_texture, mean_perimeter, mean_area,
        mean_smoothness, mean_compactness, mean_concavity,
        mean_concave_points, mean_symmetry, mean_fractal_dimension, result
    ))
    conn.commit()
    conn.close()

    return render_template('index.html', prediction=result)

@app.route('/output')
def output():
    conn = sqlite3.connect('BC.db')
    c = conn.cursor()
    c.execute('''
        SELECT user_name, mean_radius, mean_texture, mean_perimeter, mean_area,
               mean_smoothness, mean_compactness, mean_concavity,
               mean_concave_points, mean_symmetry, mean_fractal_dimension, prediction
        FROM predictions
    ''')
    records = c.fetchall()
    conn.close()

    return render_template('database.html', records=records)

if __name__ == '__main__':
    app.run(debug=True)