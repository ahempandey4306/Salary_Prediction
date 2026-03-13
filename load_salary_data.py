import pandas as pd
import os

def load_salary_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Salary Data.csv")
    
    try:
        df = pd.read_csv(csv_path)
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        print("\nDataset info:")
        print(df.info())
        
        print("\nBasic statistics:")
        print(df.describe())
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File 'Salary Data.csv' not found in {current_dir}")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    df = load_salary_data()
    
    if df is not None:
        print("\n" + "="*50)
        print("SALARY DATASET ANALYSIS")
        print("="*50)
        
        print("\nUnique values in categorical columns:")
        categorical_columns = ['Gender', 'Education Level', 'Job Title']
        
        for col in categorical_columns:
            if col in df.columns:
                print(f"\n{col}:")
                print(df[col].value_counts())
        
        print("\nSalary statistics by Education Level:")
        print(df.groupby('Education Level')['Salary'].agg(['mean', 'median', 'std', 'min', 'max']))
        
        print("\nSalary statistics by Gender:")
        print(df.groupby('Gender')['Salary'].agg(['mean', 'median', 'std', 'min', 'max']))
