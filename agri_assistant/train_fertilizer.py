import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report
import os
import numpy as np

def initialize_environment():
    """Create required folders if they don't exist"""
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def load_and_validate_data():
    """Load and validate the dataset"""
    try:
        df = pd.read_csv('data/fertilizer_data.csv')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_')
        
        # Fix common column name variations
        if 'Phosphorous' in df.columns:
            df.rename(columns={'Phosphorous': 'Phosphorus'}, inplace=True)
        
        # Check for required columns
        required_columns = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 
                            'Moisture', 'Soil_Type', 'Crop_Type', 'Fertilizer_Name']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"❌ Missing columns: {', '.join(missing_columns)}")
            return None
        
        # Clean fertilizer names
        df['Fertilizer_Name'] = df['Fertilizer_Name'].str.strip()
        
        return df
    
    except FileNotFoundError:
        print("❌ Error: data/fertilizer_data.csv not found")
        print("Please create the CSV file with fertilizer data")
        return None

def train_and_save_model(df):
    """Train the model and save it"""
    # Prepare data
    X = df.drop(columns=['Fertilizer_Name'])
    y = df['Fertilizer_Name']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Define preprocessing
    numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'pH', 'Moisture']
    categorical_features = ['Soil_Type', 'Crop_Type']
    
    preprocessor = ColumnTransformer([
        ('numeric', StandardScaler(), numeric_features),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # Create pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced'
        ))
    ])
    
    # Split data with stratification to keep all classes in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Train model
    print("⚙️ Training the fertilizer recommendation model...")
    model.fit(X_train, y_train)
    
    # Evaluate — force report to include all classes
    print("\nModel Evaluation Report:")
    print(classification_report(
        y_test, model.predict(X_test),
        labels=list(range(len(label_encoder.classes_))),
        target_names=label_encoder.classes_,
        zero_division=0
    ))
    
    # Save artifacts
    joblib.dump(model, 'models/fertilizer_model.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    print("\n✅ Model successfully saved to models/fertilizer_model.pkl")
    print("✅ Label encoder saved to models/label_encoder.pkl")

def main():
    print("Fertilizer Recommendation Model Training")
    print("--------------------------------------")
    
    # Set up environment
    initialize_environment()
    
    # Load and validate data
    df = load_and_validate_data()
    if df is None:
        return
    
    # Show class distribution
    print("\nFertilizer Type Distribution:")
    print(df['Fertilizer_Name'].value_counts())
    
    # Train and save model
    train_and_save_model(df)

if __name__ == "__main__":
    main()
