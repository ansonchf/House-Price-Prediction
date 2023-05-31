# House-Price-Prediction

This project aims to predict house prices using machine learning techniques. It involves various steps such as data cleaning, feature engineering, model selection, and ensembling.

## Kaggle Competition Details

- Competition Name: House Prices - Advanced Regression Techniques
- Competition Link: [House Price Prediction Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- Public Score: 0.12591 [Top 15%]

## Files

- `House_price_prediction.ipynb`: Jupyter Notebook containing the code for data exploration, preprocessing, feature engineering, model selection, and evaluation.
- `trained_models_full.pkl` : A file containing trained machine learning models for prediction. Unzip the rar file to load the Trained model.
- `main.py`: Python script for using the trained model to make predictions on new data.
- `main_backup.py`: A backup version of the main script in case the user cannot download the model.
- `requirements.txt`: List of required Python packages for running the code.

## House_price_prediction Contents

- Exploratory Data Analysis (EDA)
- Data Cleaning (Duplicate, Outliers, and Missing Value)
- Feature Engineering
- Feature Transformations
- Feature Selection
- Preprocessor pipeline (Encoding & Scaling)
- Target Transformation
- Model Selection
- Model Evaluation
- Hyperparameter Optimization
- Ensembling Models
- Neural Network

## Usage

To use the trained models to predict house prices, follow the instructions below:

1. Make sure you have the required Python packages installed. You can install them by running `pip install -r requirements.txt` (replace `requirements.txt` with the actual filename if different).

2. Prepare the input data:
   - Ensure that your data is in the same format as the provided test dataset (`data/test.csv`).
   - Update the filename in the `main.py` file to point to your data file.

3. Run the `main.py` file:
   - Open a terminal or command prompt and navigate to the project directory.
   - Run the command: `python main.py`.
   - Follow the prompts to enter the filename for your data.

4. Prediction:
   - The script will generate predictions for the house prices based on the trained models.
   - The predictions will be saved as a CSV file named `Prediction.csv` in the `submission` directory.

Note: If the user is unable to download the model file `trained_models_full.pkl`, `main_backup.py` script can be used as an alternative.

## Repository Structure
├── data/ # Directory containing the data files

│ └── test.csv # Test dataset file

├── submission/ # Directory for submission files

│ └── Prediction.csv # Predicted house prices file

├── main.py # Main script for data processing and prediction

├── requirements.txt # Required Python packages

├── README.md # Project README file (you are here)

├── trained_models_full.rar # unzip rar to load the Trained model
