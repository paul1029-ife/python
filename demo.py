import pandas as pd
import os
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

def download_dataset(): 
    try:
        path = kagglehub.dataset_download("srinivasav22/sales-transactions-dataset")
        training = pd.read_excel(f"{path}/Train.xlsx", nrows=200)
        
        os.makedirs("datasets", exist_ok=True)
        training.to_csv("./datasets/For_Prediction.csv", index=False)
        
        print("Dataset saved successfully to For_Prediction.csv")
        return training

    except FileNotFoundError:
        print("Error: Dataset file not found")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The dataset is empty")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None

def dataset_statistics(file_path):
    try:
        df = pd.read_csv(file_path)
        
        num_rows, num_cols = df.shape
        print(f"\nDataset Statistics:")
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_cols}")
        
        # Get unique values in Suspicious column
        suspicious_values = df['Suspicious'].nunique()
        print(f"Number of unique values in Suspicious column: {suspicious_values}")
        print(f"Unique values in Suspicious column: {df['Suspicious'].unique()}")

    except Exception as e:
        print(f"Error analyzing dataset: {str(e)}")

def perform_linear_regression(file_path):
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        
        # Convert Suspicious column to numeric
        # 'No' -> 0, 'indeterminate' -> 1
        df['Suspicious'] = df['Suspicious'].map({'No': 0, 'indeterminate': 1})
        
        # Select features (TotalSalesValue and Quantity)
        X = df[['TotalSalesValue', 'Quantity']]
        y = df['Suspicious']  # Target variable
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Round predictions to nearest integer since Suspicious is categorical
        y_pred_rounded = [round(pred) for pred in y_pred]
        
        # Print results
        print("\nSuspicious Transaction Predictions:")
        print("\nFeature Coefficients:")
        print(f"Total Sales Value coefficient: {model.coef_[0]:.4f}")
        print(f"Quantity coefficient: {model.coef_[1]:.4f}")
        
        print("\nSample of Actual vs Predicted Values:")
        print("Total Sales | Quantity | Actual | Predicted")
        print("-" * 50)
        for i in range(5):
            print(f"${X_test.iloc[i, 0]:.2f} | {X_test.iloc[i, 1]:.0f} | {y_test.iloc[i]:.0f} | {y_pred_rounded[i]:.0f}")
        
        # Calculate and print model performance
        mse = mean_squared_error(y_test, y_pred_rounded)
        accuracy = accuracy_score(y_test, y_pred_rounded)
        print(f"\nModel Performance:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Accuracy: {accuracy:.2%}")

    except Exception as e:
        print(f"Error in regression analysis: {str(e)}")

# Main execution
print("Downloading and preparing dataset...")
training_data = download_dataset()

if training_data is not None:
    dataset_statistics("./datasets/For_Prediction.csv")
    perform_linear_regression("./datasets/For_Prediction.csv")