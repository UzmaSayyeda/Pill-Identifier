
from django.shortcuts import render
import keras
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
from django.core.files.storage import default_storage



def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def results(request): 
    return render(request, 'results.html')