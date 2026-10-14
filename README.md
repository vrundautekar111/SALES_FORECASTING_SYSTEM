
# Sales Forecasting and Prediction System

## Project Overview

The Sales Forecasting and Prediction System is a
machine learning web application that analyzes historical
sales data and predicts sales for the next six months.

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- HTML5
- CSS3
- JavaScript
- Chart.js

## Main Features

1. Dark-themed sales analytics dashboard.
2. Total and average sales calculations.
3. Latest-month sales and growth indicator.
4. Random Forest regression forecasting.
5. Six-month sales predictions.
6. MAE and RMSE evaluation metrics.
7. CSV upload and validation.
8. Downloadable forecast results.
9. Responsive dashboard layout.

## Installation

Create a virtual environment:

    python -m venv .venv

Activate it in Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install -r requirements.txt

Start the application:

    python app.py

Open this address in your browser:

    http://127.0.0.1:5000

## Dataset Format

The CSV file must contain:

    date,sales

Use at least eight valid monthly observations.

## Machine Learning Method

The Random Forest Regressor uses:
- Time index
- Month number
- Previous month's sales
- Sales from two months earlier
- Previous three-month average

MAE and RMSE are used to evaluate predictions on
a time-ordered holdout period.

## Limitations

The forecast is an educational baseline. Accuracy depends
on data quality, history length, seasonality, and business
conditions. Synthetic data should not be used for real
business decisions.
