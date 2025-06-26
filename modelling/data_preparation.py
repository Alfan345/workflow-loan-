import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(data_path='../preprocessed_loan_data.csv'):
    """
    Load and prepare data for model training
    
    Args:
        data_path: Path to the preprocessed data file
        
    Returns:
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    """
    # Load the preprocessed
    df = pd.read_csv(data_path)
    
    # Split features and target
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to numpy arrays to avoid MLflow autolog issues
    X_train_scaled = np.array(X_train_scaled)
    X_test_scaled = np.array(X_test_scaled)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
