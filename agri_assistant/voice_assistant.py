import speech_recognition as sr
import pyttsx3
from assistant import AgriAssistant
from typing import Optional

class VoiceAgriAssistant:
    def __init__(self):
        self.assistant = AgriAssistant()
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        
        # Voice configuration
        self.engine.setProperty('rate', 150)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)  # Female voice

    def speak(self, text: str):
        """Convert text to speech"""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self) -> Optional[str]:
        """Listen to user voice input"""
        with sr.Microphone() as source:
            print("Listening...")
            self.recognizer.adjust_for_ambient_noise(source)
            try:
                audio = self.recognizer.listen(source, timeout=5)
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

    def handle_crop_recommendation(self):
        """Voice interface for crop recommendations"""
        self.speak("Let's get crop recommendations. I'll need some information about your soil and climate.")
        
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
                if response:
                    try:
                        value = float(response)
                        params[param] = value
                        break
                    except ValueError:
                        self.speak("Please enter a valid number.")

        result = self.assistant.get_crop_recommendation(params)
        if 'error' in result:
            self.speak(f"Sorry, there was an error: {result['error']}")
        else:
            self.speak(f"Based on these conditions, I recommend planting {result['crop']}")

    def handle_weather_query(self):
        """Voice interface for weather queries"""
        self.speak("Which city's weather would you like to know?")
        city = self.listen()
        
        if city:
            weather = self.assistant.get_weather_data(city)
            if 'error' in weather:
                self.speak(f"Sorry, {weather['error']}")
            else:
                self.speak(
                    f"In {weather['city']}, the weather is {weather['weather']} "
                    f"with a temperature of {weather['temperature']} degrees Celsius, "
                    f"humidity at {weather['humidity']} percent, "
                    f"and atmospheric pressure of {weather['pressure']} hectopascals."
                )

    def run(self):
        """Main voice assistant loop"""
        self.speak("Welcome to the Agriculture Voice Assistant. How can I help you today?")
        
        while True:
            try:
                command = self.listen()
                if not command:
                    continue
                    
                if 'exit' in command or 'quit' in command:
                    self.speak("Goodbye!")
                    break
                    
                response = self.assistant.process_query(command)
                
                if response["type"] == "crop_recommendation":
                    self.handle_crop_recommendation()
                elif response["type"] == "weather":
                    self.handle_weather_query()
                else:
                    self.speak(response["response"])
                    for option in response["options"]:
                        self.speak(option)
                        
            except KeyboardInterrupt:
                self.speak("Shutting down the assistant.")
                break

if __name__ == "__main__":
    assistant = VoiceAgriAssistant()
    assistant.run()