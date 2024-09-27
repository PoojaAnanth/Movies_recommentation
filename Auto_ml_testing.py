
import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
from sklearn.metrics import mean_squared_error

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to prepare data
def prepare_data(df):
    # Here you can implement any necessary preprocessing steps
    # For example, you might want to encode categorical variables or handle missing values
    return df

# Function to train AutoML using TPOT
def train_automl(X_train, y_train, X_test, y_test):
    # Initialize the TPOTRegressor
    tpot = TPOTRegressor(verbosity=2, generations=5, population_size=20, random_state=42)
    
    # Fit the model on the training data
    tpot.fit(X_train, y_train)
    
    # Make predictions on the test data
    predictions = tpot.predict(X_test)
    
    # Calculate mean squared error
    mse = mean_squared_error(y_test, predictions)
    
    # Export the best pipeline
    tpot.export('best_model_pipeline.py')
    
    return mse, tpot.fitted_pipeline_

def main():
    # Load the dataset
    df = load_data('data/movies_dataset.csv')
    data = prepare_data(df)

    # For the AutoML model, prepare features and target
    X = data[['User_ID', 'Movie_ID']]  # Features (you can add more features if available)
    y = data['Ratings']                 # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train AutoML model
    automl_mse, best_model = train_automl(X_train, y_train, X_test, y_test)

    # Print results
    print(f"Best AutoML Model Mean Squared Error: {automl_mse}")
    print(f"Best AutoML Pipeline:\n{best_model}")

# Check if the script is run directly
if __name__ == "__main__":
    main()

