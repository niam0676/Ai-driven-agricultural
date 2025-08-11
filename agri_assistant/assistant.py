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
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# Fertilizer mapping (update according to your dataset codes)
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
        self.crop_model = None
        self.fertilizer_model = None
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.crop_data_path = os.getenv("CROP_DATA_PATH")
        self.fertilizer_data_path = os.getenv("FERTILIZER_DATA_PATH", "data/fertilizer_data.csv")
        self.model_dir = os.getenv("MODEL_DIR", "models")
        self._initialize_models()

    def _initialize_models(self):
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
        required_cols = ["Nitrogen", "Phosphorus", "Potassium", "pH", "Moisture",
                         "Soil_Type", "Crop_Type", "Fertilizer_Name"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError("Fertilizer data is missing required columns")

        # Map fertilizer codes to names if numeric
        if pd.api.types.is_numeric_dtype(df["Fertilizer_Name"]):
            df["Fertilizer_Name"] = df["Fertilizer_Name"].map(FERTILIZER_MAP)

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

    def get_crop_recommendation(self, params: Dict[str, float]) -> Dict[str, Any]:
        if self.crop_model is None:
            return {"error": "Crop recommendation service is currently unavailable"}
        required_params = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        if not all(p in params for p in required_params):
            missing = [p for p in required_params if p not in params]
            return {"error": f"Missing parameters: {', '.join(missing)}"}
        input_data = [[params[p] for p in required_params]]
        prediction = self.crop_model.predict(input_data)[0]
        return {"crop": prediction, "parameters": params, "status": "success"}

    def get_fertilizer_recommendation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.fertilizer_model is None:
            return {"error": "Fertilizer recommendation service is currently unavailable"}
        
        required_params = ['nitrogen', 'phosphorus', 'potassium', 'ph', 'crop']
        optional_params = {'moisture': 30, 'soil_type': 'Loamy'}
        
        missing = [p for p in required_params if p not in params]
        if missing:
            return {"error": f"Missing required parameters: {', '.join(missing)}"}

        features = {
            "Nitrogen": params['nitrogen'],
            "Phosphorus": params['phosphorus'],
            "Potassium": params['potassium'],
            "pH": params['ph'],
            "Moisture": params.get('moisture', optional_params['moisture']),
            "Soil_Type": params.get('soil_type', optional_params['soil_type']),
            "Crop_Type": params['crop']
        }

        df = pd.DataFrame([features])
        prediction = self.fertilizer_model.predict(df)[0]
        proba = self.fertilizer_model.predict_proba(df)[0]
        classes = self.fertilizer_model.classes_

        # 🔹 Convert numeric predictions to names if needed
        if isinstance(prediction, (int, float)):
            prediction = FERTILIZER_MAP.get(int(prediction), prediction)

        alternatives = []
        for fert, prob in zip(classes, proba):
            if fert != prediction and prob > 0.1:
                # Convert numeric alternatives to names
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

    def get_weather_data(self, city: str) -> Dict[str, Any]:
        try:
            if not self.weather_api_key:
                return {"city": city, "weather": "Sunny", "temperature": 25, "humidity": 60,
                        "pressure": 1012, "source": "sample data"}
            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            complete_url = f"{base_url}q={city}&appid={self.weather_api_key}&units=metric"
            response = requests.get(complete_url)
            data = response.json()
            if data["cod"] != "404":
                main = data["main"]
                weather = data["weather"][0]
                return {"city": city, "weather": weather['description'], "temperature": main['temp'],
                        "humidity": main['humidity'], "pressure": main['pressure'], "source": "OpenWeatherMap"}
            else:
                return {"error": "City not found"}
        except Exception as e:
            return {"error": f"Weather API error: {str(e)}"}

    def process_query(self, query: str) -> Dict[str, Any]:
        query = query.lower()
        if 'crop' in query:
            return {"type": "crop_recommendation", "action": "request_parameters"}
        elif 'fertilizer' in query:
            return {"type": "fertilizer_recommendation", "action": "request_parameters"}
        elif 'weather' in query:
            return {"type": "weather", "action": "request_location"}
        else:
            return {"type": "general",
                    "response": "I can help with crop recommendations, fertilizer suggestions, and weather information",
                    "options": ["Ask about crop recommendations", "Ask about fertilizer recommendations", "Ask about weather in your area"]}

if __name__ == "__main__":
    assistant = AgriAssistant()
    print("Agriculture Assistant (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip().lower()
        if user_input == 'quit':
            print("Goodbye!")
            break

        response = assistant.process_query(user_input)

        if response["type"] == "crop_recommendation":
            print("\n=== Crop Recommendation ===")
            params = {
                'nitrogen': float(input("Nitrogen (N) level (0-150): ")),
                'phosphorus': float(input("Phosphorus (P) level (0-150): ")),
                'potassium': float(input("Potassium (K) level (0-150): ")),
                'temperature': float(input("Temperature (°C): ")),
                'humidity': float(input("Humidity (%): ")),
                'ph': float(input("Soil pH (0-14): ")),
                'rainfall': float(input("Rainfall (mm): "))
            }
            result = assistant.get_crop_recommendation(params)
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
            else:
                print("\n🌱 Crop Recommendation Result")
                print(f"Crop: {result['crop']}")
                for k, v in result['parameters'].items():
                    print(f"{k.title()}: {v}")

        elif response["type"] == "fertilizer_recommendation":
            print("\n=== Fertilizer Recommendation ===")
            params = {
                'crop': input("Crop name: ").strip().title(),
                'nitrogen': float(input("Nitrogen (0-200): ")),
                'phosphorus': float(input("Phosphorus (0-200): ")),
                'potassium': float(input("Potassium (0-200): ")),
                'ph': float(input("Soil pH (0-14): ")),
                'moisture': float(input("Moisture (%) [default=30]: ") or 30),
                'soil_type': input("Soil type [default=Loamy]: ").strip().title() or "Loamy"
            }
            result = assistant.get_fertilizer_recommendation(params)
            if "error" in result:
                print(f"\n❌ Error: {result['error']}")
            else:
                print("\n🌿 Fertilizer Recommendation Result")
                print(f"Fertilizer: {result['fertilizer']} (Confidence: {result['confidence']:.0%})")
                if result["alternatives"]:
                    print("\nAlternative Options:")
                    for alt in sorted(result["alternatives"], key=lambda x: x["probability"], reverse=True):
                        print(f"- {alt['fertilizer']} ({alt['probability']:.0%})")
                print("\nParameters Used:")
                for k, v in result['parameters'].items():
                    print(f"{k}: {v}")

        elif response["type"] == "weather":
            city = input("Enter city name: ").strip()
            weather = assistant.get_weather_data(city)
            if "error" in weather:
                print(f"\n❌ Error: {weather['error']}")
            else:
                print("\n🌤️ Weather Report")
                print(f"City: {weather['city'].title()}")
                print(f"Condition: {weather['weather'].title()}")
                print(f"Temperature: {weather['temperature']}°C")
                print(f"Humidity: {weather['humidity']}%")
                print(f"Pressure: {weather['pressure']} hPa")
                print(f"Source: {weather['source']}")

        else:
            print(response["response"])
            for option in response["options"]:
                print(f"- {option}")
