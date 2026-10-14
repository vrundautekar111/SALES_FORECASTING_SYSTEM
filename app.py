
from flask import Flask, render_template, request, jsonify, send_file
from pathlib import Path
import pandas as pd
import numpy as np
import io

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "sales_data.csv"

DATA_DIR.mkdir(exist_ok=True)


def create_sample_data():
    """Create 36 months of demonstration sales data."""
    rng = np.random.default_rng(42)

    dates = pd.date_range(
        start="2023-01-01",
        periods=36,
        freq="MS"
    )

    trend = np.linspace(8500, 14500, len(dates))
    seasonality = 1800 * np.sin(
        2 * np.pi * (dates.month.to_numpy() - 1) / 12
    )
    noise = rng.normal(0, 650, len(dates))

    sales = np.maximum(
        1000, trend + seasonality + noise
    )

    df = pd.DataFrame({
        "date": dates,
        "sales": np.round(sales, 2)
    })

    df.to_csv(DATA_FILE, index=False)


def load_sales_data():
    """Load, validate and aggregate sales records."""

    if not DATA_FILE.exists():
        create_sample_data()

    df = pd.read_csv(DATA_FILE)

    date_column = next(
        (
            col for col in df.columns
            if col.strip().lower()
            in ["date", "order date", "order_date", "month"]
        ),
        None
    )

    sales_column = next(
        (
            col for col in df.columns
            if col.strip().lower()
            in ["sales", "revenue", "amount", "total sales"]
        ),
        None
    )

    if date_column is None or sales_column is None:
        raise ValueError(
            "CSV must contain date and sales columns."
        )

    df = df[[date_column, sales_column]].copy()
    df.columns = ["date", "sales"]

    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    df["sales"] = pd.to_numeric(
        df["sales"].astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False),
        errors="coerce"
    )

    df = df.dropna()
    df = df.groupby("date", as_index=False)["sales"].sum()
    df = df.sort_values("date")

    if len(df) < 8:
        raise ValueError(
            "At least 8 valid sales records are required."
        )

    return df


def build_forecast(df, periods=6):
    """Train a Random Forest and predict future monthly sales."""

    monthly = (
        df.set_index("date")["sales"]
        .resample("MS")
        .sum()
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(monthly) < 8:
        raise ValueError(
            "At least 8 months of data are needed."
        )

    monthly["t"] = np.arange(len(monthly))
    monthly["month"] = monthly["date"].dt.month
    monthly["lag1"] = monthly["sales"].shift(1)
    monthly["lag2"] = monthly["sales"].shift(2)

    monthly["rolling3"] = (
        monthly["sales"].shift(1)
        .rolling(3)
        .mean()
    )

    features = ["t", "month", "lag1", "lag2", "rolling3"]
    training_data = monthly.dropna().copy()

    if len(training_data) < 5:
        raise ValueError("Not enough data to train the model.")

    # Evaluate on later months, preserving time order.
    split = max(3, int(len(training_data) * 0.8))
    split = min(split, len(training_data) - 1)

    evaluation_model = RandomForestRegressor(
        n_estimators=250,
        random_state=42
    )

    evaluation_model.fit(
        training_data.iloc[:split][features],
        training_data.iloc[:split]["sales"]
    )

    actual = training_data.iloc[split:]["sales"]
    predicted = evaluation_model.predict(
        training_data.iloc[split:][features]
    )

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))

    # Train the final model using all available observations.
    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42
    )

    model.fit(
        training_data[features],
        training_data["sales"]
    )

    sales_history = monthly["sales"].astype(float).tolist()
    last_date = monthly["date"].max()
    forecast_rows = []

    for step in range(1, periods + 1):
        next_date = last_date + pd.offsets.MonthBegin(step)

        feature_row = pd.DataFrame([{
            "t": len(monthly) + step - 1,
            "month": next_date.month,
            "lag1": sales_history[-1],
            "lag2": sales_history[-2],
            "rolling3": float(np.mean(sales_history[-3:]))
        }])

        prediction = max(
            0.0,
            float(model.predict(feature_row[features])[0])
        )

        forecast_rows.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "sales": round(prediction, 2)
        })

        # Use each forecast as an input for the next month.
        sales_history.append(prediction)

    return monthly, forecast_rows, mae, rmse


@app.route("/")
def index():
    try:
        df = load_sales_data()
        history, forecast, mae, rmse = build_forecast(df)

        total_sales = float(df["sales"].sum())
        average_sales = float(df["sales"].mean())
        latest_sales = float(df.iloc[-1]["sales"])

        previous_sales = float(df.iloc[-2]["sales"])
        growth = (
            (latest_sales - previous_sales)
            / max(abs(previous_sales), 1)
        ) * 100

        return render_template(
            "index.html",
            history_dates=[
                date.strftime("%Y-%m-%d")
                for date in history["date"]
            ],
            history_sales=[
                round(float(value), 2)
                for value in history["sales"]
            ],
            forecast=forecast,
            total=round(total_sales, 2),
            avg=round(average_sales, 2),
            latest=round(latest_sales, 2),
            growth=round(growth, 1),
            mae=round(mae, 2),
            rmse=round(rmse, 2),
            rows=len(df),
            error=None
        )

    except Exception as exc:
        return render_template(
            "index.html",
            history_dates=[],
            history_sales=[],
            forecast=[],
            total=0,
            avg=0,
            latest=0,
            growth=0,
            mae=None,
            rmse=None,
            rows=0,
            error=str(exc)
        )


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")

    if not file or not file.filename.lower().endswith(".csv"):
        return jsonify({
            "ok": False,
            "message": "Please select a CSV file."
        }), 400

    try:
        df = pd.read_csv(file)

        # Validate the uploaded file before replacing existing data.
        date_col = next(
            (c for c in df.columns
             if c.strip().lower() in
             ["date", "order date", "order_date", "month"]),
            None
        )

        sales_col = next(
            (c for c in df.columns
             if c.strip().lower() in
             ["sales", "revenue", "amount", "total sales"]),
            None
        )

        if date_col is None or sales_col is None:
            raise ValueError(
                "CSV must contain date and sales columns."
            )

        test_df = df[[date_col, sales_col]].copy()
        test_df.columns = ["date", "sales"]
        test_df["date"] = pd.to_datetime(
            test_df["date"], errors="coerce"
        )
        test_df["sales"] = pd.to_numeric(
            test_df["sales"].astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False),
            errors="coerce"
        )
        test_df = test_df.dropna()

        if len(test_df) < 8:
            raise ValueError(
                "Upload at least 8 valid sales records."
            )

        # Validate that the data can actually be forecast.
        monthly_test = (
            test_df.groupby("date")["sales"].sum()
            .resample("MS").sum()
        )

        if len(monthly_test) < 8:
            raise ValueError(
                "Please provide at least 8 months of sales data."
            )

        # Keep the original file if validation fails.
        df.to_csv(DATA_FILE, index=False)

        return jsonify({
            "ok": True,
            "message": "Data uploaded successfully. Refreshing dashboard..."
        })

    except Exception as exc:
        return jsonify({
            "ok": False,
            "message": str(exc)
        }), 400


@app.route("/download-forecast")
def download_forecast():
    df = load_sales_data()
    _, forecast, _, _ = build_forecast(df)

    output = io.BytesIO()
    pd.DataFrame(forecast).to_csv(output, index=False)
    output.seek(0)

    return send_file(
        output,
        mimetype="text/csv",
        as_attachment=True,
        download_name="sales_forecast.csv"
    )


if __name__ == "__main__":
    if not DATA_FILE.exists():
        create_sample_data()

    app.run(debug=True, port=5000)
