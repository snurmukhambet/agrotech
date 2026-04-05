# CropYield AI — Innovative AI Challenge 2024

**Challenge 1: Developing AI Models to Increase Agricultural Productivity**

A machine learning system for predicting crop yield (kg/ha) based on agronomic and environmental features, developed as part of the Innovative AI Challenge 2024.

---

## Authors

**Students:** Sayan Nurmukhambet, Aidar Amangeldy, Bahyt Madenali

---

## Abstract

Accurate prediction of agricultural productivity is a critical problem in food security research and precision farming. This project develops a supervised regression model based on the Random Forest algorithm to estimate crop yield (kg/ha) from a set of environmental and agronomic features including annual rainfall, soil classification, irrigated area, crop type, and geographic region. The system is deployed as an interactive web application allowing real-time inference and exploratory data analysis.

---

## Application Features

| Module | Description |
|--------|-------------|
| Yield Prediction | User-defined parameter input with instant model inference and yield distribution context |
| Data Insights | Interactive visualizations: grouped bar charts, scatter plots, box plots, correlation heatmaps, temporal trends |
| Model Analysis | Feature importance ranking, hyperparameter summary, actual vs. predicted diagnostic plot |

**Model performance:** Random Forest Regressor, R² approximately 95%, RMSE below 400 kg/ha.

---

## Deployment (Streamlit Community Cloud — Free)

1. Push this repository to your GitHub account.
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and authenticate via GitHub.
3. Select **New app**, choose the repository, and set `app.py` as the entry point.
4. Click **Deploy**.

The application will be available at a public URL of the form `https://username-appname.streamlit.app`. No server configuration or payment is required.

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/snurmukhambet/agrotech.git
cd agrotech

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
```

The application will be available at `http://localhost:8501`.

---

## Repository Structure

```
crop-yield-ai/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

Note: Training data is generated synthetically within the application using a fixed random seed for full reproducibility. No external CSV files are required. To substitute real data, replace the `generate_data()` function in `app.py` with a `pd.read_csv()` call pointing to the target dataset.

---

## Dataset Description

| Feature | Type | Description |
|---------|------|-------------|
| `Year` | Numeric | Year of observation |
| `State` | Categorical | Indian state or administrative region |
| `Crop_Type` | Categorical | Crop species (Rice, Wheat, Bajra, Maize, Cotton, Sugarcane) |
| `Rainfall` | Numeric | Mean annual precipitation (mm) |
| `Soil_Type` | Categorical | Soil classification (Loamy, Sandy, Clay, Black, Red) |
| `Irrigation_Area` | Numeric | Irrigated land area (thousand hectares) |
| `Crop_Yield (kg/ha)` | Numeric | Target variable — crop productivity per unit area |

---

## Methodology

Categorical features (State, Crop_Type, Soil_Type) are encoded via label encoding. Numerical features (Rainfall, Irrigation_Area) are standardized using z-score normalization. The Random Forest Regressor is trained on an 80/20 stratified split with the following hyperparameters:

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| max_depth | 20 |
| max_features | sqrt |
| min_samples_split | 5 |
| min_samples_leaf | 2 |

---

## Technology Stack

- **Streamlit** — web application framework
- **scikit-learn** — model training and evaluation
- **Plotly** — interactive visualizations
- **Seaborn / Matplotlib** — static statistical graphics
- **Pandas / NumPy** — data manipulation and numerical computation