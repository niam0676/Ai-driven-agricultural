import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import requests
import os
import joblib
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Fertilizer mapping
FERTILIZER_MAP = {
    1: "Urea",
    2: "DAP",
    3: "MOP",
    4: "Ammonium Sulphate",
    5: "SSP",
    6: "NPK 20:20:20"
}

class AgriAssistant:
    def __init__(self):
        """Initialize with all APIs and models"""
        self.crop_model = None
        self.fertilizer_model = None
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.crop_data_path = os.getenv("CROP_DATA_PATH")
        self.fertilizer_data_path = os.getenv("FERTILIZER_DATA_PATH", "data/fertilizer_data.csv")
        self.model_dir = os.getenv("MODEL_DIR", "models")
        self._initialize_models()

    def _initialize_models(self):
        """Load and prepare all recommendation models"""
        self._initialize_crop_model()
        self._initialize_fertilizer_model()

    def _initialize_crop_model(self):
        try:
            if os.path.exists(self.crop_data_path):
                crop_data = pd.read_csv(self.crop_data_path)
                required_cols = ['nitrogen', 'phosphorus', 'potassium', 'temperature',
                                 'humidity', 'ph', 'rainfall', 'crop']
                if all(col in crop_data.columns for col in required_cols):
                    X = crop_data[required_cols[:-1]]
                    y = crop_data['crop']
                    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
                    self.crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
                    self.crop_model.fit(X_train, y_train)
                    print("✅ Crop recommendation model loaded successfully")
                else:
                    raise ValueError("Crop data CSV is missing required columns")
            else:
                raise FileNotFoundError(f"Crop data file not found at {self.crop_data_path}")
        except Exception as e:
            print(f"❌ Error initializing crop model: {str(e)}")
            self.crop_model = None

    def _initialize_fertilizer_model(self):
        try:
            model_path = os.path.join(self.model_dir, "fertilizer_model.pkl")
            if os.path.exists(model_path):
                self.fertilizer_model = joblib.load(model_path)
                print("✅ Fertilizer recommendation model loaded from cache")
            elif os.path.exists(self.fertilizer_data_path):
                print("⚙️ Training fertilizer recommendation model...")
                self._train_fertilizer_model()
            else:
                print(f"⚠️ Fertilizer data not found at {self.fertilizer_data_path}")
        except Exception as e:
            print(f"❌ Error initializing fertilizer model: {str(e)}")
            self.fertilizer_model = None

    def _train_fertilizer_model(self):
        df = pd.read_csv(self.fertilizer_data_path)
        if df["Fertilizer_Name"].dtype in ['int64', 'float64']:
            df["Fertilizer_Name"] = df["Fertilizer_Name"].map(FERTILIZER_MAP)

        required_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture", "Soil_Type", "Crop_Type", "Fertilizer_Name"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Fertilizer data is missing required columns")

        X = df.drop(columns=["Fertilizer_Name"])
        y = df["Fertilizer_Name"]

        numeric_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture"]
        categorical_cols = ["Soil_Type", "Crop_Type"]

        numeric_transformer = Pipeline([("scaler", StandardScaler())])
        categorical_transformer = Pipeline([("encoder", OneHotEncoder(handle_unknown="ignore"))])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
        ])

        pipeline.fit(X, y)
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(pipeline, os.path.join(self.model_dir, "fertilizer_model.pkl"))
        self.fertilizer_model = pipeline
        print("✅ Fertilizer model trained & saved!")

    def get_fertilizer_by_location(self, crop: str, city: str) -> Dict[str, Any]:
        """Get fertilizer recommendation using weather API and crop-specific soil needs"""
        weather = self.get_weather_data(city)
        if "error" in weather:
            return weather

        try:
            df = pd.read_csv(self.fertilizer_data_path)
            crop_data = df[df['Crop_Type'].str.lower() == crop.lower()].mean(numeric_only=True)
            
            params = {
                'Crop_Type': crop.title(),
                'Nitrogen': crop_data.get('Nitrogen', 80),
                'Phosphorus': crop_data.get('Phosphorus', 40),
                'Potassium': crop_data.get('Potassium', 35),
                'pH': crop_data.get('pH', 6.5),
                'Moisture': weather.get('humidity', 30),
                'Soil_Type': 'Loamy',
                'temperature': weather.get('temperature', 25)
            }
            
            return self.get_fertilizer_recommendation(params)
            
        except Exception as e:
            return {"error": f"Error processing crop data: {str(e)}"}

    def get_crop_by_city(self, city: str) -> Dict[str, Any]:
        weather = self.get_weather_data(city)
        if "error" in weather:
            return weather

        df = pd.read_csv(self.fertilizer_data_path)
        region_data = df.mean(numeric_only=True)
        
        params = {
            "nitrogen": region_data.get('Nitrogen', 80),
            "phosphorus": region_data.get('Phosphorus', 40),
            "potassium": region_data.get('Potassium', 35),
            "temperature": weather["temperature"],
            "humidity": weather["humidity"],
            "ph": region_data.get('pH', 6.5),
            "rainfall": weather.get('rain', 100)
        }
        return self.get_crop_recommendation(params)

    def get_crop_recommendation(self, params: Dict[str, float]) -> Dict[str, Any]:
        if self.crop_model is None:
            return {"error": "Crop recommendation service is currently unavailable"}
        try:
            cols = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
            input_df = pd.DataFrame([params], columns=cols)
            prediction = self.crop_model.predict(input_df)[0]
            return {"crop": prediction, "parameters": params, "status": "success"}
        except Exception as e:
            return {"error": f"Error making recommendation: {str(e)}"}

    def get_fertilizer_recommendation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.fertilizer_model is None:
            return {"error": "Fertilizer recommendation service is currently unavailable"}
        try:
            features = {
                "Nitrogen": params.get('Nitrogen', params.get('nitrogen', 80)),
                "Phosphorus": params.get('Phosphorus', params.get('phosphorus', 40)),
                "Potassium": params.get('Potassium', params.get('potassium', 35)),
                "pH": params.get('pH', params.get('ph', 6.5)),
                "Moisture": params.get('Moisture', params.get('moisture', 30)),
                "Soil_Type": params.get('Soil_Type', params.get('soil_type', 'Loamy')),
                "Crop_Type": params.get('Crop_Type', params.get('crop'))
            }
            
            if not features["Crop_Type"]:
                return {"error": "Crop type is required"}
                
            cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture", "Soil_Type", "Crop_Type"]
            input_df = pd.DataFrame([features], columns=cols)
            prediction = self.fertilizer_model.predict(input_df)[0]
            if isinstance(prediction, (int, float)):
                prediction = FERTILIZER_MAP.get(int(prediction), prediction)

            proba = self.fertilizer_model.predict_proba(input_df)[0]
            classes = self.fertilizer_model.classes_
            alternatives = []
            for fert, prob in zip(classes, proba):
                if fert != prediction and prob > 0.1:
                    if isinstance(fert, (int, float)):
                        fert = FERTILIZER_MAP.get(int(fert), fert)
                    alternatives.append({"fertilizer": fert, "probability": float(prob)})

            return {
                "fertilizer": prediction,
                "confidence": float(max(proba)),
                "alternatives": alternatives,
                "parameters": features,
                "status": "success"
            }
        except Exception as e:
            return {"error": f"Error making fertilizer recommendation: {str(e)}"}

    def get_weather_data(self, city: str) -> Dict[str, Any]:
        try:
            if not self.weather_api_key:
                return {"city": city, "weather": "Sunny", "temperature": 25, "humidity": 60, "pressure": 1012, "source": "sample data"}
            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            complete_url = f"{base_url}q={city}&appid={self.weather_api_key}&units=metric"
            response = requests.get(complete_url)
            data = response.json()
            if data["cod"] != "404":
                main = data["main"]
                weather = data["weather"][0]
                return {
                    "city": city,
                    "weather": weather['description'],
                    "temperature": main['temp'],
                    "humidity": main['humidity'],
                    "pressure": main['pressure'],
                    "source": "OpenWeatherMap"
                }
            else:
                return {"error": "City not found"}
        except Exception as e:
            return {"error": f"Weather API error: {str(e)}"}

    def process_query(self, query: str) -> Dict[str, Any]:
        query = query.lower()
        if 'crop' in query:
            return {"type": "crop_by_city", "action": "request_city"}
        elif 'fertilizer' in query:
            return {"type": "fertilizer_recommendation", "action": "request_parameters"}
        elif 'weather' in query:
            return {"type": "weather", "action": "request_location"}
        else:
            return {"type": "general", "response": "I can help with crop recommendations, fertilizer suggestions, and weather information"}

if __name__ == "__main__":
    assistant = AgriAssistant()
    print("\n🌾 Agriculture Assistant (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip().lower()
        if user_input == 'quit':
            print("Goodbye!")
            break
        response = assistant.process_query(user_input)
        if response["type"] == "crop_by_city":
            city = input("Enter city name: ").strip()
            result = assistant.get_crop_by_city(city)
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
            else:
                print(f"\n🌱 Recommended crop for {city.title()}: {result['crop']}")
        elif response["type"] == "fertilizer_recommendation":
            print("\n=== 🌿 Fertilizer Recommendation ===")
            city = input("Enter city name: ").strip()
            crop = input("Enter crop name: ").strip().title()
            result = assistant.get_fertilizer_by_location(crop, city)
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
            else:
                print(f"\n🌿 For {crop} in {city.title()}, I recommend: {result['fertilizer']} (Confidence: {result['confidence']:.0%})")
                if result["alternatives"]:
                    print("\nAlternative Options:")
                    for alt in sorted(result["alternatives"], key=lambda x: x["probability"], reverse=True):
                        print(f"- {alt['fertilizer']} ({alt['probability']:.0%})")
        elif response["type"] == "weather":
            city = input("Enter city name: ").strip()
            weather = assistant.get_weather_data(city)
            if "error" in weather:
                print(f"\n❌ Error: {weather['error']}")
            else:
                print(f"\n🌤️ Weather in {weather['city']} ({weather['source']}): {weather['weather'].title()}, {weather['temperature']}°C, {weather['humidity']}% humidity")
        else:
            print(response["response"])