# House-Price-Prediction

This project aims to predict house prices using machine learning techniques. It involves various steps such as data cleaning, feature engineering, model selection, and ensembling.

## Contents

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
   - Ensure that your data is in the same format as the provided training dataset (`data/train.csv`).
   - Create a CSV file with the test data in the same format, excluding the `Id` column.
   - Update the filename in the `main.py` file to point to your test data file.

3. Run the `main.py` file:
   - Open a terminal or command prompt and navigate to the project directory.
   - Run the command: `python main.py`.
   - Follow the prompts to enter the filename for your test data.

4. Prediction:
   - The script will generate predictions for the house prices based on the trained models.
   - The predictions will be saved as a CSV file named `Prediction.csv` in the `submission` directory.

## Repository Structure
├── data/ # Directory containing the data files
│ └── train.csv # Training dataset file
├── submission/ # Directory for submission files
│ └── Prediction.csv # Predicted house prices file
├── main.py # Main script for data processing and prediction
├── requirements.txt # Required Python packages
├── README.md # Project README file (you are here)
