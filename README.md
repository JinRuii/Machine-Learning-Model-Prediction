# 🌆 Machine Learning Model Prediction for Urban Spatial Analysis

![Python](https://img.shields.io/badge/Python-3.8-blue.svg)
![scikit-learn](https://img.shields.io/badge/Scikit--Learn-0.24.2-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-3.2.1-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.2.4-yellow.svg)

## 🔍 Project Overview

This repository provides a suite of machine learning models for predicting target labels based on various urban and environmental features. By examining patterns in data such as housing prices, pollution levels, and accessibility to amenities, these models reveal insights that support smarter city planning and environmental management. Whether you're a data scientist, urban planner, or policymaker, this framework empowers you to explore how different factors influence urban areas and to make informed, data-driven decisions.

The implemented models include:
- 🌲 **Decision Trees**
- 🌳 **Random Forest**
- 🔍 **K-Nearest Neighbors (KNN)**
- 📈 **Gradient Boosting Decision Trees (GBDT)**
- ⚡ **Extreme Gradient Boosting (XGBoost)**
- 🌟 **Light Gradient Boosting Machine (LightGBM)**

> **Note**: LightGBM is highlighted here as an example due to its efficiency and effectiveness in handling large datasets, making it well-suited for urban spatial data analysis.

## 🏙️ Application in Urban Planning

Machine learning has become an invaluable tool for urban planning, allowing us to analyze spatial relationships and make predictions on critical urban factors. By applying these models to urban data, we can:

- 🏘️ **Identify Housing Trends**: Pinpoint areas with rising housing demand or pricing changes.
- 🌳 **Optimize Green Space**: Identify regions lacking public green spaces, walkability, or recreational facilities.
- 🚍 **Enhance Public Transport Access**: Assess and improve access to transportation hubs like metro and bus stations.
- 🌍 **Mitigate Environmental Risks**: Predict areas with higher pollution risks, enabling proactive environmental management and public health interventions.

This project is built to support urban planners, researchers, and decision-makers in creating sustainable, livable cities. By understanding urban patterns and spatial distributions of social and environmental factors, we can work toward a better, more balanced urban future.

---

## 🌟 LightGBM Model Overview  (example here)

The **LightGBM model** is a central part of this repository, focusing on efficient training and evaluation across large datasets.

### Key Steps for LightGBM:
1. **Data Loading & Preprocessing**: Load features from `H8-16_variables.csv` and map labels to numeric values.
2. **Feature Scaling**: Normalize data using Min-Max scaling.
3. **Data Splitting**: Split data into training, validation, and test sets.
4. **Training & Evaluation**: Train on the training set and log performance metrics (accuracy, precision, recall, F1 score) for all data splits.
5. **Model Saving**: Save the trained model as `trained_model_LightGBM.pkl` for future predictions.

### 🔑 Features Used
Key features include:
- **Housing Price (HOP)** 🏠
- **Population Density (POD)** 👥
- **Distance to Bus/Metro Stations** 🚉
- **Points of Interest (POIs)** (shopping, medical, educational) 🏥
- **Environmental Factors** (PM2.5, PM10, CO levels) 🌫️

### 📊 Evaluation Metrics
Each model is evaluated on:
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

## ⚙️ Hyperparameters

Default hyperparameters for LightGBM:
```python
hyperparams = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
