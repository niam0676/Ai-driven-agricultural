import re
import speech_recognition as sr
import pyttsx3
from assistant import AgriAssistant
from typing import Optional


def parse_float(text: str) -> Optional[float]:
    """Extract number from spoken text like 'six point five' → 6.5."""
    if not text:
        return None
    text = text.lower().replace("point", ".")
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        try:
            return float(numbers[0])
        except ValueError:
            return None
    return None


class VoiceAgriAssistant:
    def __init__(self):
        self.assistant = AgriAssistant()
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()

        # Voice config
        self.engine.setProperty('rate', 150)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)  # Female

    def speak(self, text: str):
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self) -> Optional[str]:
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=6)
                text = self.recognizer.recognize_google(audio)
                print(f"You: {text}")
                return text.lower()
            except sr.WaitTimeoutError:
                self.speak("I didn't hear anything. Please try again.")
                return None
            except sr.UnknownValueError:
                self.speak("Sorry, I didn't understand that.")
                return None
            except sr.RequestError:
                self.speak("There was an error with the speech service.")
                return None

    def handle_crop_by_city(self):
        """Get crop recommendation from city."""
        self.speak("Tell me your city name.")
        city = self.listen()
        if not city:
            return
        result = self.assistant.get_crop_by_city(city.title())
        if "error" in result:
            self.speak(f"Sorry, {result['error']}")
        else:
            self.speak(f"For {city.title()}, I recommend growing {result['crop']}.")

    def handle_crop_recommendation(self):
        """Manual crop recommendation with soil and weather params."""
        self.speak("Let's get crop recommendations. I will ask for some soil and climate values.")
        params = {}
        questions = {
            'nitrogen': "What is the nitrogen level in your soil, between 0 and 150?",
            'phosphorus': "What is the phosphorus level, between 0 and 150?",
            'potassium': "What is the potassium level, between 0 and 150?",
            'temperature': "What is the current temperature in Celsius?",
            'humidity': "What is the humidity percentage?",
            'ph': "What is your soil pH value, between 0 and 14?",
            'rainfall': "What is the recent rainfall in millimeters?"
        }

        for param, question in questions.items():
            while True:
                self.speak(question)
                response = self.listen()
                value = parse_float(response)
                if value is not None:
                    params[param] = value
                    break
                else:
                    self.speak("Please say a valid number.")

        result = self.assistant.get_crop_recommendation(params)
        if 'error' in result:
            self.speak(f"Sorry, there was an error: {result['error']}")
        else:
            self.speak(f"Based on these conditions, I recommend planting {result['crop']}.")

    def handle_fertilizer_recommendation(self):
        """Simplified fertilizer recommendation using only crop and location"""
        self.speak("For fertilizer recommendation, please tell me the crop name.")
        crop = None
        while not crop:
            crop = self.listen()
            if not crop:
                self.speak("I didn't catch the crop name. Please try again.")

        self.speak(f"Got it. Now please tell me the city or region for {crop}.")
        city = None
        while not city:
            city = self.listen()
            if not city:
                self.speak("I didn't catch the location. Please try again.")

        # Get recommendation using weather API and dataset
        result = self.assistant.get_fertilizer_by_location(crop, city.title())
        
        if "error" in result:
            self.speak(f"Sorry, {result['error']}")
        else:
            self.speak(f"For {crop.title()} in {city.title()}, I recommend {result['fertilizer']}.")
            if result.get("alternatives"):
                alts = ", ".join(a["fertilizer"] for a in result["alternatives"])
                self.speak(f"Alternative options: {alts}")

    def handle_weather_query(self):
        """Weather by city."""
        self.speak("Which city's weather would you like to know?")
        city = self.listen()
        if city:
            weather = self.assistant.get_weather_data(city)
            if 'error' in weather:
                self.speak(f"Sorry, {weather['error']}")
            else:
                self.speak(
                    f"In {weather['city']}, the weather is {weather['weather']}, "
                    f"temperature {weather['temperature']} degrees Celsius, "
                    f"humidity {weather['humidity']} percent."
                )

    def run(self):
        """Main loop."""
        self.speak("Welcome to the Agriculture Voice Assistant. You can ask for crop, fertilizer, or weather.")
        while True:
            try:
                command = self.listen()
                if not command:
                    continue
                if 'quit' in command or 'exit' in command:
                    self.speak("Goodbye!")
                    break
                elif 'crop by city' in command:
                    self.handle_crop_by_city()
                elif 'crop' in command:
                    self.handle_crop_recommendation()
                elif 'fertilizer' in command:
                    self.handle_fertilizer_recommendation()
                elif 'weather' in command:
                    self.handle_weather_query()
                else:
                    self.speak("I can help with crop, fertilizer, or weather. Please say one of these.")
            except KeyboardInterrupt:
                self.speak("Shutting down the assistant.")
                break


if __name__ == "__main__":
    assistant = VoiceAgriAssistant()
    assistant.run()