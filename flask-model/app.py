from flask import Flask, request, render_template, flash,  jsonify
from flask_cors import CORS
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

#SQL imports
# from sqlalchemy import create_engine, Column, Integer, String, Date
# from sqlalchemy.orm import sessionmaker
# from sqlalchemy.ext.declarative import declarative_base

# SQLite database path for pill predicitons
# db_path = 'sqlite:///Data/pill_predicitions.db'
# engine = create_engine(db_path)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5000"}})
CORS(app, resources={r"/api/*": {"origins": "http://127.0.0.1:5500"}})


csv_path = 'Data/rximagesAll.csv'


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
model = tf.keras.models.load_model('MobileNet_02.keras')
print('model successfully loaded!')

start = [0]
passed = [0]
pack = [[]]
num = [0]

 

# with open('pills20.csv', 'r') as file:
#     reader = csv.reader(file)
#     pill_table = dict()
#     for i, row in enumerate(reader):
#         if i == 0:
#             name = ''
#             continue
#         else:
#             name = row[1].strip()
#         pill_table[name] = [
#             {'Drug Class': str(row[2])},
#             {'Generic Name': str(row[3])},
#             {'Pill Name': str(row[4])},
#             {'Uses': str(row[5])}
         
#         ]

@app.route("/")
@app.route("/home")
def index():
    return render_template('home.html')

@app.route("/predict")
def predict():
    return render_template('predict.html')

@app.route("/chart")
def chart():
    # df = pd.read_sql('SELECT * FROM rximagesAll', conn = engine.raw_connection())
    return render_template('charts.html')

@app.route("/credits")
def credit():
    return render_template('credits.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.getlist("img")
    for f in file:
        filename = secure_filename(str(num[0] + 500) + '.jpg')
        num[0] += 1
        name = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print('save name', name)
        f.save(name)

    pack[0] = []
    
    return render_template('predict.html', img=file)

@app.route('/results')
def results():
    # pack = []
    print('total image', num[0])
        
    for i in range(start[0], num[0]):
        pa = dict()
        x = dict()

        filename = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
        print('image filepath', filename)
        pred_img = filename
        pred_img = image.load_img(pred_img, target_size=(224, 224))
        pred_img = image.img_to_array(pred_img)
        pred_img = np.expand_dims(pred_img, axis=0)
        pred_img = pred_img / 255.

        pred = model.predict(pred_img)
        print("Pred")
        print(pred)

        if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and \
                math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
            pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])

        top = pred.argsort()[0][-3:]
        # label.sort()
        _true = label[top[2]]
        _trues = label[top[2]]
        x[_true] = float("{:.2f}".format(pred[0][top[2]] * 100))
        print(x[_true])
        x[label[top[1]]] = float("{:.2f}".format(pred[0][top[1]] * 100))
        print(x[label[top[1]]])
        x[label[top[0]]] = float("{:.2f}".format(pred[0][top[0]] * 100))

        pa['result'] = x
        print(x)
        pa['image'] = f'{UPLOAD_FOLDER}/{i + 500}.jpg'
       
        pack[0].append(pa)
        passed[0] += 1

    start[0] = passed[0]
    print('successfully packed')
    # compute the average source of calories

    return render_template('results.html', pack=pack[0], prediction = _trues)



@app.route('/update', methods=['POST'])
def update():
    return render_template('home.html', img='static/P2.jpg')


if __name__ == "__main__":
    import click

    @click.command()
    @click.option('--debug', is_flag=True)
    @click.option('--threaded', is_flag=True)
    @click.argument('HOST', default='127.0.0.1')
    @click.argument('PORT', default=5000, type=int)
    def run(debug, threaded, host, port):
        """
        This function handles command line parameters.
        Run the server using
            python server.py
        Show the help text using
            python server.py --help
        """
        HOST, PORT = host, port
        app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)
    run()