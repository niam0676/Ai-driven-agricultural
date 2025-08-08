import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

class AgriAssistant:
    def __init__(self):
        """Initialize with all APIs and models"""
        self.crop_model = None
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        self.crop_data_path = os.getenv("CROP_DATA_PATH")
        self._initialize_models()

    def _initialize_models(self):
        """Load and prepare crop recommendation model"""
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

    def get_crop_recommendation(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Get crop recommendation with input validation"""
        if self.crop_model is None:
            return {"error": "Crop recommendation service is currently unavailable"}
        
        required_params = ['nitrogen', 'phosphorus', 'potassium',
                           'temperature', 'humidity', 'ph', 'rainfall']
        
        try:
            if not all(p in params for p in required_params):
                missing = [p for p in required_params if p not in params]
                return {"error": f"Missing parameters: {', '.join(missing)}"}
            
            input_data = [[params[p] for p in required_params]]
            prediction = self.crop_model.predict(input_data)[0]
            
            return {
                "crop": prediction,
                "parameters": params,
                "status": "success"
            }
        except Exception as e:
            return {"error": f"Error making recommendation: {str(e)}"}

    def get_weather_data(self, city: str) -> Dict[str, Any]:
        """Get weather data for a city"""
        try:
            if not self.weather_api_key:
                return {
                    "city": city,
                    "weather": "Sunny",
                    "temperature": 25,
                    "humidity": 60,
                    "pressure": 1012,
                    "source": "sample data"
                }
                
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
        """Main method to process all types of queries"""
        query = query.lower()
        
        if 'crop' in query or 'recommend' in query:
            return {"type": "crop_recommendation", "action": "request_parameters"}
        elif 'weather' in query:
            return {"type": "weather", "action": "request_location"}
        else:
            return {
                "type": "general",
                "response": "I can help with crop recommendations and weather information",
                "options": [
                    "Ask about crop recommendations",
                    "Ask about weather in your area"
                ]
            }

if __name__ == "__main__":
    assistant = AgriAssistant()
    print("Agriculture Assistant (type 'quit' to exit)\n")
    
    while True:
        user_input = input("You: ").lower()
        
        if user_input == 'quit':
            print("Goodbye!")
            break
            
        response = assistant.process_query(user_input)
        
        if response["type"] == "crop_recommendation":
            print("\n=== Crop Recommendation ===")
            print("Please enter the following parameters:")
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
            print(f"\n🌱 Recommended crop: {result['crop']}")
            
        elif response["type"] == "weather":
            city = input("Enter city name: ")
            weather = assistant.get_weather_data(city)
            print(f"\nWeather in {weather['city']}:")
            print(f"Condition: {weather['weather']}")
            print(f"Temperature: {weather['temperature']}°C")
            print(f"Humidity: {weather['humidity']}%")
            print(f"Pressure: {weather['pressure']} hPa")
            
        else:
            print(response["response"])
            for option in response["options"]:
                print(f"- {option}")
