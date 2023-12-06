# Pill-Identifier
The purpose of this project is to create an imaging tool that will be used to identify different types of pills

# Using Machine Learning to Identify Pills
![Pills](<flask-model/static/images/bg2.jpg>)

Feel free to view our [website](https/"our link here") 

# Table of Contents

1. [Machine Learning Modelling on Malaria Cell Image Recognition](#machine-learning-modelling-on-malaria-cell-image-recognition)
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

## Introduction

The primary goal of this project is to utilize machine learning to analyze and identify pill images based off image files submitted by a user to our web application. To achieve this objective, we have developed a web application that integrates an optimized machine learning model. This application is designed for use by "put info here". The primary intention behind this application is to "Put info here"

## Our team:
* TJ 
* Uzma
* Jonathan 
* Evan
* Yelemi

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

![EDA Header](<Dataset/EDA/EDA header.png>)
"insert our image here"
* [Jupyter Notebook Code for EDA](<EDA.ipynb>) 

### Model Analysis

* Model 1
* Model 2
* Model 3


### Data Integration and Export

The results of the various analyses were integrated into 4 DataFrames. 2 for the training data: ```Dataset/eda_train_infect.csv```, ```Dataset/eda_train_uninfect.csv```, and two for the testing data: ```Dataset/eda_test_infect.csv```, ```Dataset/eda_test_uninfect.csv``` for further analysis and reference for useage on the ML model.


**Building the Machine Learning**
We tried a few different machine learning models to figure out the best accuracy for our end goal. 

### Machine Learning Models:
EDA:
| Model | Accuracy|
|-----:|---------------|
| Random Forest (RF) |   83%  |
| Random Forest + hyperparameter |  89%  |
| RF + Gradient Boosting       | 83%      |
| Linear Regression Model      | 32%      |
| SVC Model                    | 79%      |
| SVC+Hyp                      | 85%      |
| SVC + PCA                    | 79%      |
| SVC+RF+NN                    | 81%      |
| Decision Tree                | 74%     |

IMAGE:
| Model | Accuracy|
|-----:|---------------|
| CNN  |   94%  |
| K-NN  |  60%|
| Xception  |   80%  |
| Xception Optimized  |  79%|


**A) [CNN](<optimized_MLmodel.ipynb>):**

The CNN model was designed for the classification of cell images into two categories: uninfected (0) and infected (1). It's aimed at assisting in the automated detection of infected cells, a task of significance in the detection of Malaria.

- **Training Dataset:**
  - Dataset Size: 1750 cell images.
  - Features: Each row in the dataset represents an image, with each pixel of the image treated as a feature.
  - Target Variable: The "Target" column indicates the class label, where 0 represents uninfected cells, and 1 represents infected cells.

- **Data Preprocessing:**
  - Image Resizing: All images were resized to a consistent size (not specified in the provided information) to ensure uniform input dimensions for the CNN.
  - Normalization: Pixel values were scaled to a range of [0, 1] by dividing by the maximum pixel value (e.g., 255 for 8-bit images). This standardization helps improve convergence during training.

- **Model Architecture:**
  - The CNN model architecture used for cell image classification is as follows:
    - Input Layer: Accepts images with dimensions (32, 25, 25, 3)
    - Convolutional Layers: Three convolutional layers were employed with varying numbers of filters and filter sizes.
    - Max-Pooling Layers: Max-pooling layers followed each convolutional layer to reduce spatial dimensions.
    - Flatten Layer: The output from the final max-pooling layer was flattened into a 1D vector of length 16.
    - Dense Layers: Two dense (fully connected) layers were used.
    - Dropout Layer: A dropout layer with a dropout rate of 0.5 was added after the first dense layer to prevent overfitting.

- **Model Training:**
  - Loss Function: Binary cross-entropy.
  - Optimizer: Adam
  - Batch Size: 32
  - Epochs: 50
  - Validation Split: 20% of the training data was used for validation during training to monitor model performance.

- **Model Evaluation:**
  - The model's performance was evaluated using common binary classification metrics, including:
    - Accuracy: Measures the overall correctness of predictions. For this model, the accuracy was 94%.

**B) [Random Forest](<Various_models.ipynb>):**

The EDA data was analyzed by a Random Forest model to predict if a cell was infected or not. The Random Forest model was tuned with hyperparameters, and the important features were identified.

- **Testing Data:**
  - The testing dataset of 750 images that were not seen by the model was used to test the dataset. The model was used to predict infected and uninfected cells.

- **Results:**
  - The Random forest with hyperparameter tuning gave an accuracy of 89%.

**C) [Xception](<xception.ipynb>):**
 
- **Model Architecture:**
      - Input Layer: Accepts images with dimensions (25, 25, 3)
      - Base Model: Xception with pre-trained weights
      - Input Layer: Accepts images with variable dimensions
      - Convolutional and Separable Convolutional Layers
      - Batch Normalization Layers
      - Activation Layers
      - Max-Pooling Layers
      - Global Average Pooling Layer
      - Two Dense (Fully Connected) Layers
    - Output Layer: Dense layer with 1 neuron and sigmoid activation
    - Freeze some layers in the base model to prevent them from being trained.

- **Model Training:**
  - Loss Function: Binary cross-entropy.
  - Optimizer: Adam
  - Batch Size: 32
  - Epochs: 20 for base model and 10 for top model
  - Validation Split: 20% of the training data was used for validation during training to monitor model performance.

- **Model Evaluation:**
  - The model's performance was evaluated using common binary classification metrics, including:
    - Accuracy: For base model was 78-80% while for top model was 77%. 

- This is a pre-trained model on the popular image dataset called `imagenet`. We built our base model using the pre-trained model and then added a layer of our own testing and training dataset to see how it performs. We got an accuracy of 79% over 20 epochs.
- Model was fine-tuned by adding creating `top_model` on top of the `base_model` by feeding the output from the base model to the top model. It was interesting to notice that the accuracy did not change much but the loss had a significant difference. 

- **Model Summary:**
    - Model architecture with detailed layer information for top model.
    - Total parameters: 22,960,681 (87.59 MB)
    - Trainable parameters: 2,099,201 (8.01 MB)
    - Non-trainable parameters: 20,861,480 (79.58 MB)
    - The graphs in `xception.ipynb` depict that the data was overfitting at certain points but the validation set performed better than the training data consistently. 
    
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

