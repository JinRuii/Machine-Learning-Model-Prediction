import torch
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier  # Import KNN model
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv
import time  # Used to calculate training time

# 1. Set input and output file directories
input_dir = "./data/H8-16"  # Input file directory
output_dir = "./output/H8-16"  # Output file directory
os.makedirs(output_dir, exist_ok=True)  # Automatically create output directory if it doesn't exist

# Label value mapping
label_mapping = {'none': 0, 'low': 1, 'relatively low': 2, 'relatively high': 3, 'high': 4}

# Train KNN model and save results
def train_KNN_model(features, labels, hyperparams, output_dir):
    start_time = time.time()  # Record start time

    # Split data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define KNN model
    knn_model = KNeighborsClassifier(
        n_neighbors=hyperparams['n_neighbors'],
        weights=hyperparams['weights'],
        algorithm=hyperparams['algorithm'],
        leaf_size=hyperparams['leaf_size'],
        p=hyperparams['p'],
        metric=hyperparams['metric'],
        n_jobs=-1
    )

    # Train model
    knn_model.fit(X_train, y_train)

    # Evaluate performance on training, validation, and test sets
    y_train_pred = knn_model.predict(X_train)
    y_val_pred = knn_model.predict(X_val)
    y_test_pred = knn_model.predict(X_test)

    # Calculate evaluation metrics
    def evaluate_metrics(y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        return accuracy, f1, precision, recall

    train_acc, train_f1, train_precision, train_recall = evaluate_metrics(y_train, y_train_pred)
    val_acc, val_f1, val_precision, val_recall = evaluate_metrics(y_val, y_val_pred)
    test_acc, test_f1, test_precision, test_recall = evaluate_metrics(y_test, y_test_pred)

    # Print results
    print(f"Training Set: Accuracy = {train_acc:.4f}, Precision = {train_precision:.4f}, Recall = {train_recall:.4f}, F1 Score = {train_f1:.4f}")
    print(f"Validation Set: Accuracy = {val_acc:.4f}, Precision = {val_precision:.4f}, Recall = {val_recall:.4f}, F1 Score = {val_f1:.4f}")
    print(f"Test Set: Accuracy = {test_acc:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}, F1 Score = {test_f1:.4f}")

    # Save training log
    log_file = os.path.join(output_dir, 'training_logs_KNN.csv')
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['set', 'accuracy', 'precision', 'recall', 'f1_score'])
        writer.writerow(['train', train_acc, train_precision, train_recall, train_f1])
        writer.writerow(['val', val_acc, val_precision, val_recall, val_f1])
        writer.writerow(['test', test_acc, test_precision, test_recall, test_f1])

    # Record end time and calculate training duration
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

    # Save model
    model_save_path = os.path.join(output_dir, 'trained_model_KNN.pkl')
    with open(model_save_path, 'wb') as f:
        import pickle
        pickle.dump(knn_model, f)

    # Print final test set results
    print(f"Final Results - Test Set: Accuracy = {test_acc:.4f}, Precision = {test_precision:.4f}, Recall = {test_recall:.4f}, F1 Score = {test_f1:.4f}")

# Load data and initiate training
if __name__ == '__main__':
    # Load feature and label data
    variables_file = os.path.join(input_dir, 'H8-16_variables.csv')  # Path to feature file
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

    # Load labels and map to numerical values
    labels = variables_df['TD_label'].map(label_mapping).values

    # Set KNN hyperparameters
    hyperparams = {
        'n_neighbors': 5,         # Value of K
        'weights': 'uniform',     # Weight function
        'algorithm': 'auto',      # Nearest neighbor search algorithm
        'leaf_size': 30,          # Leaf size for tree construction
        'p': 2,                   # Distance metric, 2 means Euclidean distance
        'metric': 'minkowski'     # Method for distance measurement
    }

    # Train KNN model
    train_KNN_model(features, labels, hyperparams, output_dir)

    print("KNN model training complete!")
