Agriculture Assistant
Introduction
The Agriculture Assistant is a Python-based project designed to help farmers and users decide which crop is best to grow based on soil nutrients and environmental conditions. It can also provide current weather information for any location using an online weather API.

Components Used
Software Components
Python 3 – Programming language used for the entire project.

Pandas – For reading and managing crop dataset.

Scikit-learn – For training the crop recommendation machine learning model.

Requests – For getting weather data from the OpenWeatherMap API.

Python-dotenv – For storing and loading API keys securely from a .env file.

Hardware Requirements
A computer or laptop with Python installed.

Internet connection (for fetching weather data).

Dataset Used
The dataset contains the following columns:

Nitrogen (N) – Level of nitrogen in the soil.

Phosphorus (P) – Level of phosphorus in the soil.

Potassium (K) – Level of potassium in the soil.

Temperature – Temperature of the location in Celsius.

Humidity – Humidity percentage in the air.

pH – pH value of the soil.

Rainfall – Rainfall amount in millimeters.

Crop – The crop that is suitable for the given conditions.

Working Procedure
The program loads the crop dataset and trains a Random Forest Classifier machine learning model using the given soil and weather parameters.

The user can choose between two main features:

Crop Recommendation – User enters soil and climate values, and the program predicts the best crop to grow.

Weather Information – User enters a city name, and the program fetches live weather details from the OpenWeatherMap API.

The system processes the inputs and displays the output in a simple text format.

Steps to Use
Keep your crop dataset ready in .csv format.

Store your OpenWeatherMap API key in a .env file along with the dataset path.

Run the program using Python.

Choose the required option (crop recommendation or weather information).

Enter the required details when prompted.

View the recommendation or weather data.

Example Usage
If you choose Crop Recommendation, enter soil nutrient levels, temperature, humidity, pH, and rainfall values, and the program will suggest a crop.

If you choose Weather Information, enter your city name, and the program will display the weather condition, temperature, humidity, and pressure.

Applications
Helps farmers plan their crops more effectively.

Useful for agricultural students and researchers.

Can be integrated with larger agricultural management systems.