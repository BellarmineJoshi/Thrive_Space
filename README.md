# Streamlit Q-Learning App

## Overview
This project is a **mental wellness application** built using **Streamlit**, integrating **Q-learning** to enhance user experiences based on sentiment analysis. The app dynamically generates recommendations based on users' emotional states.

## Features
- **User Input Form**: Collects emotional state data.
- **Sentiment Analysis**: Uses TextBlob to analyze the user's input.
- **Q-Learning Algorithm**: Determines the best action (recommendation) to improve the user's sentiment score.
- **Dynamic Content Display**: Adjusts UI elements based on sentiment.
- **Streamlit UI**: Interactive and user-friendly interface for easy accessibility.

## Technologies Used
- **Python**
- **Streamlit** (for UI)
- **TextBlob** (for sentiment analysis)
- **Q-learning** (for reinforcement learning implementation)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/streamlit-qlearning-app.git
   cd streamlit-qlearning-app
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Open the app in your browser.
2. Enter your current emotional state.
3. The app performs sentiment analysis and applies Q-learning to suggest improvements.
4. View the updated sentiment score and recommended actions.

## Future Enhancements
- Improve reinforcement learning efficiency.
- Enhance visualization of sentiment trends.
- Deploy the app for public access.


