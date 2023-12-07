# Pill-Identifier
The purpose of this project is to create an imaging tool that will be used to identify different types of pills

# Using Machine Learning to Identify Pills
![Pills](<flask-model/static/images/bg2.jpg>)

Feel free to view our [website](https/"our link here") 

# Table of Contents

1. [Pill Image recognition with Machine Learning](#pill-image-recognition-with-machine-learning) 
   1. [Introduction](#introduction)
   2. [Team](#our-team)
   3. [Programs Utilized](#programs-utilized)
   4. [Dataset used](#dataset-used)
   5. [Project Management](#project-management)
   6. [Data Cleaning and Exploratory Data Analysis](#data-cleaning-and-exploratory-data-analysis)
      1. [Data Preprocessing](#data-preprocessing)
      2. [Notes on the Data Cleaning/Preprocessing Process](#notes-on-the-data-cleaningpreprocessing-process)
   7. [Exploratory Data Analysis](#exploratory-data-analysis)
   8. [Model Analysis](#model-analysisgit )
   10. [Data Integration and Export](#data-integration-and-export)
   11. [Machine Learning Models](#machine-learning-models)
2. [FLASK](#flask)
   1. [Key Libraries Used](#key-libraries-used)
   2. [Backend Processes & AWS Integration](#backend-processes--aws-integration)
   3. [User Interaction & Data Handling](#user-interaction--data-handling)
   4. [API Creation](#api-creation)
3. [Front End Development for User Integration](#front-end-development-for-user-integration)
   1. [Front-end](#front-end)
   2. [Visualization](#visualization)
   3. [Styling & Fonts](#styling--fonts)
   4. [Backend & Storage](#backend--storage)
   5. [User Interaction](#user-interaction)
4. [Presentation](#presentation)
5. [References](#references)


## Pill Image recognition with Machine Learning


## Introduction

The primary goal of this project is to utilize machine learning to analyze and identify pill images based off image files submitted by a user to our web application. To achieve this objective, we have developed a web application that integrates an optimized machine learning model. This application is designed for use by "put info here". The primary intention behind this application is to "Put info here"

## Our team:
* TJ Davis
* Uzma Sayyeda
* Jonathan Carmichael
* Evan Michigan
* Yelemi Lee

## Programs Utilized:

### Backend 
* Python: Matplotlib, Numpy, Pandas, Sklearn, Tensorflow, PIL, OS, CV2, boto3, Keras, Random, CV2, SplitFolders
* Flask
* SQLite
* S3.Bucket
* CSV Files

### Frontend
* HTML/CSS: 
* Javascript: Plotly
* 
 
### Other
* GitHub
* Discord
* Slack
* 
* 

## Dataset used:
* Downloadable from: [Pill Images](https://www.nlm.nih.gov/databases/download/pill_image.html)
* 

## Project Management:



## Data Cleaning and Exploratory Data Analysis

### Data Preprocessing 
* [Jupyter Notebook code for Data Preprocessing](<split_data_mobilenet.ipynb>)


#### Notes on the Data Cleaning/Preprocessing Process
* For this model we utilized an S3 bucket for data storage. 
* We downloaded the image files from the bucket and selected which consisted of 552 files
* In Pyton using the splitfolders function we then split that into a training dataset and a validation dataset
at a ratio=(.8, 0.1,0.1)
* We preproccessed and scaled the images to 224/224/1
* The training directory used 437 images belonging to 23 classes
* The validation directory used 46 images belonging to 23 classes




## Exploratory Data Analysis

![EDA Header](<flask-model/local_drug_directory/Amoxicillin 500 mg/Amoxicillin_500_1024.jpg>)

* [Jupyter Notebook Code for EDA](<flask-model/EDA.ipynb>) 

### Model Analysis

* Model 1
* Model 2
* Model 3


### Data Integration and Export

The results of the various analyses were integrated into 4 DataFrames. 2 for the training data: ```Dataset/eda_train_infect.csv```, ```Dataset/eda_train_uninfect.csv```, and two for the testing data: ```Dataset/eda_test_infect.csv```, ```Dataset/eda_test_uninfect.csv``` for further analysis and reference for useage on the ML model.


**Building the Machine Learning**
We tried a few different machine learning models to figure out the best accuracy for our end goal. 

### Machine Learning Models:

| Model | Accuracy|
|-----:|---------------|
| VGG16 |   83%  |
| RestNet50 |  8%  |
| MobileNet       | 100%      |



**A) [VGG16](<flask-model/optimization_models.ipynb>):**
The first model we used was VGG16 (Visual Geometry Group 16). VGG16 is a convolutional neural network (CNN) architecture designed for image classification. It consists of 16 layers, including 13 convolutional layers and 3 fully connected layers. Our model was designed predicting the correct medication name an dosage based off an image of the medication. It constructs a new neural network model using the Sequential API from Keras, where the base model is added as the first layer, followed by a Global Average Pooling layer, and finally, a Dense layer with 23 units and softmax activation for multi-class classification.

- **Training Dataset:**
  - Dataset Size: 437 pill images.
  - Features: Each row in the dataset represents an image, with each pixel of the image treated as a feature.
  - Target Variable: The "Target" column indicates the class label, where 0 represents uninfected cells, and 1 represents infected cells.

- **Data Preprocessing:**
  - Image Resizing: All images were resized to a consistent size  to ensure uniform input dimensions for the CNN.
  

- **Model Architecture:**

  - The same Model Architecture was used for each model.

    - Base Model Integration:
      The architecture starts with a pre-existing neural network model referred to as the "base model.
    - Sequential Model Construction:
      - For each tuple (name, base_model) in the base_models list:
       -  A new Sequential model is created.
          The base model is added as the first layer of the Sequential model.
    - Layers Added to Sequential Model:
        - Following the base model, the Sequential model has the following  layers added:
          - Global Average Pooling 2D layer: This layer reduces the spatial dimensions of the data to a single value per feature map using global average pooling.
          - Dense layer with 23 units and softmax activation: This layer is the output layer with 23 units, indicating a multi-class classification task with 23 classes. The softmax activation is commonly used for multi-class classification.
    - Model Compilation:
        - The model is compiled using the Adam optimizer, categorical crossentropy loss function (suitable for multi-class classification problems), and accuracy as the metric for evaluation.
    - Model Training:
        - The model is trained on a training data generator (train_generator) for 2 epochs.
    - Model Evaluation:
        - After training, the model is evaluated using a validation data generator (validation_generator), and the loss and accuracy are computed.
    - DataFrame Creation:
        - The results, including the model name, accuracy, and loss, are stored in lists (names, accuracy, and loss), and a pandas DataFrame (df) is created from these lists.

- **Model Training:**
  - Loss Function: Categorical-Crossentropy.
  - Optimizer: Adam
  - Epochs: 2
  - Validation Split: 20% of the training data was used for validation during training to monitor model performance.

- **Model Evaluation:**
  - The model's performance was evaluated using common binary classification metrics, including:
    - Accuracy: Measures the overall correctness of predictions. For this model, the accuracy was 83%.

**B) [RestNet50](<flask-model/optimization_models.ipynb>):**

The ResNet-50 (Residual Network with 50 layers) is a convolutional neural network (CNN) architecture that was introduced by researchers at Microsoft Research in the paper titled "Deep Residual Learning for Image Recognition" in 2015. It is part of the ResNet family, which is known for its ability to train very deep networks by introducing residual blocks.

- **Model Architecture:**
      - Same as above

- **Model Training:**
      - Same as above

- **Model Evaluation:**
  - The model's performance was evaluated using common binary classification metrics, including:
    - Accuracy: For base model was 8%. This Model performed the worst.

**C) [MobileNet](<flask-model/optimization_models.ipynb>):**

MobileNet is a family of lightweight neural network architectures designed for mobile and edge devices with limited computational resources. MobileNet models are often employed for various computer vision tasks, including image classification, object detection, and image segmentation on mobile and embedded devices. They are popular choices for applications such as real-time image processing, augmented reality, and mobile applications.
 
- **Model Architecture:**
      - Same as above

- **Model Training:**
      - Same as above

- **Model Evaluation:**
  - The model's performance was evaluated using common binary classification metrics, including:
  - Accuracy for MobilNet model was 100%. It was the best performing model.


- **Model Summary:**
    - Model architecture with detailed layer information for top model.
    - Total parameters: 14,726,847 (56.18 MB)
    - Trainable parameters: 11,799 (46.09 KB)
    - Non-trainable parameters: 14,714,688 (56.13 MB)
        
## FLASK
### Key Libraries Used:

| Type | Library|
|-----:|---------------|
| Data Handling & Processing|   Numpy, Pandas |
| Web Framework |  Flask  |
| Storage & AWS Interaction|  Boto3   |
|  File Handling & Compression|  Zipfile, IO   |
| Randomization| Random  |
|  Database & ORM| SQLAlchemy, csv  |

### Backend Processes & AWS Integration:
* To interact with our AWS storage, we generated pre-signed URLs from our S3 bucket name and key. This provides an API for Flask to retrieve image files.
 * This is publicly available data so we made the S3 bucket public as well as we were not concerned with security for this application.
### User Interaction & Data Handling:
* Flask plays a pivotal role in capturing user input data from our web game, which is temporarily stored in a global variable.
* 
### API Creation:
* Using Flask, we've set up API routes that output data in JSON format.
  
**These intricacies, woven together, create a robust and interactive platform tailored to our users' needs.**

* To run Flask: [app.py](<flask-model/app.py>)
* SQLite Database: [pill_predictions.db](<flask-model/pill_predictions.db>)
* SQL Database Generation: [sqlite.ipynb](<flask-model/sqlite.ipynb>)

## Front End Development for User Integration
Pluggins used? 

### Front-end: 
* HTML and CSS have been employed to design the visuals and effects.
### Visualization: 
* Describe here
### Styling & Fonts: 
* Describe here
### Backend & Storage:
* With Flask serving as our backend framework, we're efficiently reading data from our database.
* AWS S3 has been our choice for storage. It allows us to select image names from the database and subsequently extract and display the relevant image files on the website.
### User Interaction: 
* Describe here
**By seamlessly integrating these tools, we've been able to craft a dynamic and interactive platform for our users.**
 
  


##### Presentation

  "Insert link to presentation slides here"

##### References
*
*
*
*
* 

