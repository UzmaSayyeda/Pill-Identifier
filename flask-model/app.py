# imports
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import sqlalchemy as sql

import pandas as pd
import math
import os
import numpy as np
from keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename
import tensorflow as tf

import atexit
import shutil
import sqlite3

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5000"}})
CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})

# paths for sqlite 
db_url = 'sqlite:///drugs.db'
engine = sql.create_engine(db_url)

# path for upload folder to store images
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# define label meaning
label = ['Amoxicillin 500 mg',
         'Apixaban 2.5 mg',
         'Aprepitant 80 mg',
         'Atomoxetine 25 mg',
         'Calcitriol 0.00025',
         'Prasugrel 10 MG',
         'Ramipril 5 MG',
         'Saxagliptin 5 MG',
         'Sitagliptin 50 MG',
         'Tadalafil 5 MG',
         'carvedilol 3.125',
         'celecoxib 200',
         'duloxetine 30',
         'eltrombopag 25',
         'metformin_500',
         'montelukast-10',
         'mycophenolate-250',
         'omeprazole_40',
         'oseltamivir-45',
         'pantaprazole-40',
         'pitavastatin_1',
         'prednisone_5',
         'sertraline_25']

# Loading the best saved model to make predictions.
tf.keras.backend.clear_session()
model = tf.keras.models.load_model('MobileNet.keras')
print('model successfully loaded!')

start = 0
passed = 0
num = 0

# app.routes
@app.route("/")
@app.route("/home")
def index():
    return render_template('home.html')

# for api+sql based plotly charts
@app.route("/data")
def get_data():
    conn = sqlite3.connect("drugs.db")
    cursor = conn.cursor()
    query = "SELECT * FROM drug_effects"
    data_two = cursor.execute(query).fetchall()


    drug_data = []
    for row in data_two:
        row_dict = {}
        for i, column in enumerate(cursor.description):
            row_dict[column[0]] = row[i]
        drug_data.append(row_dict)

    conn.close()
    return jsonify(drug_data)
                
@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route("/chart")
def chart():
    return render_template('charts.html')

@app.route("/credits")
def credit():
    return render_template('credits.html')

# saves images to upload folder to predict
@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.getlist("img")
    for f in file:
        global num
        filename = secure_filename(str(num + 500) + '.jpg')
        num += 1
        name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('save name', name)
        f.save(name)
    
    return render_template('predict.html', img=file)

# predict function
@app.route('/results')
def results():
    
    global start
    global passed
    
    print('total image', num)
    
    for i in range(start, num):
        pa = dict()
        x = dict()

        # preprocess the predict image
        filename = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
        print('image filepath', filename)
        pred_img = image.load_img(filename, target_size=(224, 224))
        pred_img = image.img_to_array(pred_img)
        pred_img = np.expand_dims(pred_img, axis=0) / 255.

        # passing it to model to predict
        pred = model.predict(pred_img)
        print("Pred")
        print(pred)

        # Error handling for if model returns NaNs
        if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and \
                math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
                    
            # Arbitrarily set to list of probabilities. Should probably reset to reflect training data
            pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])

        # sorting the results
        top = pred.argsort()[0][-3:]
        
        _true = label[top[2]]
        _trues = label[top[2]]
        
        x[_true] = float("{:.2f}".format(pred[0][top[2]] * 100))
        x[label[top[1]]] = float("{:.2f}".format(pred[0][top[1]] * 100))
        x[label[top[0]]] = float("{:.2f}".format(pred[0][top[0]] * 100))

        pa['result'] = x
        print(x)
        pa['image'] = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
       
        passed += 1

    start = passed         
  

    return render_template('results.html', pack=[pa], prediction = _trues)


@app.route('/update', methods=['POST'])
def update():
    return render_template('home.html')

# clears out upload after quitting the app
def clearUpload(exception=None):
    global UPLOAD_FOLDER
    shutil.rmtree(UPLOAD_FOLDER)
    os.mkdir(UPLOAD_FOLDER)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

atexit.register(clearUpload)

if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='127.0.0.1')
    @click.argument('PORT', default=5000, type=int)
    def run(debug, threaded, host, port):
      
        HOST, PORT = host, port
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
    run()