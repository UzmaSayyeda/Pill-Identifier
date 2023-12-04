from flask import Flask, request, render_template
from flask import Flask,flash, request, render_template
import csv
import math
import os
import numpy as np
from keras.preprocessing import image
from tensorflow.python.keras.models import load_model
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.layers import BatchNormalization
import pandas as pd

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# define label meaning
label = ['Amoxicillin 500 MG',
         'apixaban 2.5 MG',
         'aprepitant 80 MG',
         'Atomoxetine 25 MG',
         'benzonatate 100 MG',
         'Calcitriol 0.00025 MG',
         'carvedilol 3.125 MG',
         'celecoxib 200 MG',
         'duloxetine 30 MG',
         'eltrombopag 25 MG',
         'montelukast 10 MG',
         'mycophenolate mofetil 250 MG',
         'Oseltamivir 45 MG',
         'pantoprazole 40 MG',
         'pitavastatin 1 MG',
         'prasugrel 10 MG',
         'Ramipril 5 MG',
         'saxagliptin 5 MG',
         'sitagliptin 50 MG',
         'tadalafil 5 MG']

# Loading the best saved model to make predictions.
tf.keras.backend.clear_session()
model = tf.keras.models.load_model('MobileNet_02.keras')
print('model successfully loaded!')

start = [0]
passed = [0]
pack = [[]]
num = [0]

 

with open('pills20.csv', 'r') as file:
    reader = csv.reader(file)
    pill_table = dict()
    for i, row in enumerate(reader):
        if i == 0:
            name = ''
            continue
        else:
            name = row[1].strip()
        pill_table[name] = [
            {'Drug Class': str(row[2])},
            {'Generic Name': str(row[3])},
            {'Pill Name': str(row[4])},
            {'Uses': str(row[5])}
         
        ]

@app.route("/")
@app.route("/index")
def index():
    return render_template('home.html')

@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route("/results")
def results():
    # if request.method == "POST":
    #     file = request.files["files"]
    #     if file:
    #         filename = secure_filename(file.filename)
    #         file.save(filename)
    #     pred_img = image.load_img(filename, target_size=(224, 224))
    #     pred_img = image.img_to_array(pred_img)
    #     pred_img = np.expand_dims(pred_img, axis=0)
    #     pred_img = pred_img / 255.
    #     pred = model.predict(pred_img)
        
    #     if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and \
    #             math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
    #         pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])
            
    #     top = pred.argsort()[0][-3:]
    #     label.sort()
    #     _true = label[top[2]]
    #     _trues = label[top[2]]

    return render_template('results.html')

@app.route('/charts')
def chart():
    return render_template('charts.html')

@app.route('/update', methods=['POST'])
def update():
    return render_template('home.html', img='static/P2.jpg')


if __name__ == "__main__":
    app.run(debug=True)