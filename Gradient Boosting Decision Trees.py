import torch
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier  # Import Gradient Boosting Decision Tree model
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv
import time  # For measuring training time

# 1. Set input and output directories
input_dir = "./data/H8-16"  # Relative path for input directory
output_dir = "./output/H8-16"  # Relative path for output directory
os.makedirs(output_dir, exist_ok=True)  # Automatically create output directory if it doesn't exist

# Label value mapping
label_mapping = {'none': 0, 'low': 1, 'relatively low': 2, 'relatively high': 3, 'high': 4}

# Train Gradient Boosting Decision Tree (GBDT) model and save results
def train_GBDT_model(features, labels, hyperparams, output_dir):
    start_time = time.time()  # Record start time

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define the GBDT model with hyperparameters
    gbdt_model = GradientBoostingClassifier(
        n_estimators=hyperparams['n_estimators'],
        learning_rate=hyperparams['learning_rate'],
        max_depth=hyperparams['max_depth'],
        max_features=hyperparams['max_features'],
        min_samples_leaf=hyperparams['min_samples_leaf'],
        min_samples_split=hyperparams['min_samples_split'],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Train the model
    gbdt_model.fit(X_train, y_train)

    # Evaluate the performance on training, validation, and test sets
    y_train_pred = gbdt_model.predict(X_train)
    y_val_pred = gbdt_model.predict(X_val)
    y_test_pred = gbdt_model.predict(X_test)

    # Function to calculate evaluation metrics
    def evaluate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        return accuracy, f1, precision, recall

    # Calculate metrics for each dataset
    train_acc, train_f1, train_precision, train_recall = evaluate_metrics(y_train, y_train_pred)
    val_acc, val_f1, val_precision, val_recall = evaluate_metrics(y_val, y_val_pred)
    test_acc, test_f1, test_precision, test_recall = evaluate_metrics(y_test, y_test_pred)

    # Print the results
    print(f"Training Set: Accuracy = {train_acc:.4f}, Precision = {train_precision:.4f}, Recall = {train_recall:.4f}, F1 Score = {train_f1:.4f}")
    print(f"Validation Set: Accuracy = {val_acc:.4f}, Precision = {val_precision:.4f}, Recall = {val_recall:.4f}, F1 Score = {val_f1:.4f}")
    print(f"Test Set: Accuracy = {test_acc:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}, F1 Score = {test_f1:.4f}")

    # Save training logs
    log_file = os.path.join(output_dir, 'training_logs_GBDT.csv')
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['set', 'accuracy', 'precision', 'recall', 'f1_score'])
        writer.writerow(['train', train_acc, train_precision, train_recall, train_f1])
        writer.writerow(['val', val_acc, val_precision, val_recall, val_f1])
        writer.writerow(['test', test_acc, test_precision, test_recall, test_f1])

    # Calculate and display training duration
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total training time: {total_time:.2f} seconds")

    # Save the trained model
    model_save_path = os.path.join(output_dir, 'trained_model_GBDT.pkl')
    with open(model_save_path, 'wb') as f:
        import pickle
        pickle.dump(gbdt_model, f)

    # Print final results for the test set
    print(f"Final Results - Test Set: Accuracy = {test_acc:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}, F1 Score = {test_f1:.4f}")

# Load data and initiate training
if __name__ == '__main__':
    # Load feature and label data
    variables_file = os.path.join(input_dir, 'H8-16_variables.csv')  # Path to your feature file
    variables_df = pd.read_csv(variables_file)

    # Extract features and normalize
    feature_names = [
        'HOP',  # Housing price, measured in yuan per square meter
        'POD',  # Population density, measured in people per square meter
        'DIS_BUS',  # Distance to the nearest bus station (poi - point of interest), measured in meters
        'DIS_MTR',  # Distance to the nearest metro station (poi), measured in meters
        'POI_COM',  # Number of company POIs (points of interest) within the area
        'POI_SHOP',  # Number of shopping POIs within the area
        'POI_SCE',  # Number of scenic spot POIs within the area
        'POI_EDU',  # Number of educational POIs within the area
        'POI_MED',  # Number of medical POIs within the area
        'PR',  # Plot ratio (building area * number of floors (height/3.5m) / area)
        'OPEN',  # Sky openness ratio from street view images; if value is -999, street view data is not available
        'CAR',  # Car presence ratio in street view images
        'GREN',  # Green view index (greenness) in street view images
        'ENCL',  # Enclosure rate in street view images
        'WAL',  # Walkability index in street view images
        'IMA',  # Imageability index in street view images
        'COMP',  # Complexity or diversity in street view images
        'PM2_5',  # Concentration of PM2.5 (particulate matter), measured in μg/m³ per hour per day
        'PM10',  # Concentration of PM10 (particulate matter), measured in μg/m³ per hour per day
        'CO'  # Carbon monoxide concentration, measured in μg/m³ per hour per day
    ]
    features = variables_df[feature_names].values
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # Load labels and apply numerical mapping
    labels = variables_df['TD_label'].map(label_mapping).values

    # Set GBDT hyperparameters
    hyperparams = {
        'n_estimators': 5000,     # Number of trees
        'learning_rate': 0.01,    # Learning rate
        'max_depth': 15,          # Maximum depth of each tree
        'max_features': 'sqrt',   # Maximum features considered per tree
        'min_samples_leaf': 10,   # Minimum samples per leaf
        'min_samples_split': 10,  # Minimum samples required to split a node
    }

    # Train the GBDT model
    train_GBDT_model(features, labels, hyperparams, output_dir)

    print("GBDT model training completed!")
