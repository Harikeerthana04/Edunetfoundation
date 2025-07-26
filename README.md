# Farm Irrigation System - Machine Learning Project

This repository contains a machine learning project aimed at predicting irrigation needs for different parcels based on sensor data. The project utilizes a RandomForest Classifier in a multi-output setting to classify the irrigation status of three distinct parcels.

## Project Files

* `Irrigation_System.ipynb`: The main Jupyter Notebook containing the Python code for data loading, preprocessing, model training, evaluation, and visualization.
* `irrigation_machine.csv`: The dataset used for training and evaluating the model. It contains sensor readings and the target irrigation status for multiple parcels.
* `Farm_Irrigation_System.pkl`: The trained machine learning model, saved using `joblib`, ready for deployment to make predictions on new sensor data.

## Overview

The goal of this project is to build an intelligent irrigation system that can decide whether to irrigate specific parcels of land based on various sensor inputs. By analyzing historical sensor data and corresponding irrigation statuses, the model learns the patterns to make accurate predictions.

## Data

The `irrigation_machine.csv` dataset comprises:
* **Sensor Readings (sensor_0 to sensor_19):** These columns represent input features from various sensors monitoring environmental or soil conditions.
* **Parcel Status (parcel_0, parcel_1, parcel_2):** These are the target variables, indicating the irrigation status (e.g., 0 for no irrigation, 1 for irrigation) for three different parcels.

## Key Libraries Used

The `Irrigation_System.ipynb` notebook uses the following Python libraries:
* `pandas`: For data manipulation and analysis.
* `matplotlib.pyplot` & `seaborn`: For data visualization and plotting charts.
* `sklearn.model_selection`: For splitting data into training and testing sets.
* `sklearn.ensemble.RandomForestClassifier`: The base estimator for the machine learning model.
* `sklearn.multioutput.MultiOutputClassifier`: To handle the multi-label classification problem (predicting irrigation for multiple parcels simultaneously).
* `sklearn.metrics`: For evaluating model performance (e.g., `classification_report`, `ConfusionMatrixDisplay`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`).
* `sklearn.preprocessing.MinMaxScaler`: For scaling numerical features.
* `joblib`: For saving and loading the trained machine learning model.
* `numpy`: For numerical operations.

## Visualizations

The notebook includes several visualizations to aid in understanding the data and evaluating the model:

* **Data Exploration:**
    * Histograms showing the distribution of scaled sensor data.
    * A Correlation Heatmap illustrating the relationships between all sensors and parcel statuses.
    * Count Plots displaying the distribution of irrigation statuses for each parcel.
* **Model Evaluation:**
    * Confusion Matrices for each parcel, providing a detailed breakdown of correct and incorrect classifications.
    * A Bar Chart comparing key performance metrics (Accuracy, Precision, Recall, F1-Score) across all parcels.
