import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropYield AI · Innovative AI Challenge 2024",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Hero banner */
.hero {
    background: #f5f5f5;
    border-top: 3px solid #333;
    padding: 24px 0;
    margin-bottom: 24px;
    color: #333;
}
.hero h1 { color: #333; font-size: 2rem; margin: 0 0 4px 0; font-weight: 600; }
.hero h2 { color: #666; font-size: 0.95rem; margin: 4px 0; font-weight: 400; }
.hero p { color: #666; font-size: 0.95rem; margin: 4px 0 0 0; }

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 16px;
    text-align: center;
}
.metric-card .value {
    font-size: 1.8rem;
    font-weight: 600;
    color: #333;
}
.metric-card .label { font-size: 0.8rem; color: #999; margin-top: 6px; }

/* Prediction result */
.prediction-box {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 24px;
    text-align: center;
    margin-top: 16px;
}
.prediction-box .yield-value {
    font-size: 2.2rem;
    font-weight: 600;
    color: #333;
}
.prediction-box .unit { font-size: 1rem; color: #666; }

/* Section headers */
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #333;
    margin: 20px 0 12px 0;
    padding: 0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: white;
}
</style>
""", unsafe_allow_html=True)


# ─── Data generation (synthetic, reproducible) ────────────────────────────────
@st.cache_data
def generate_data(n=5000, seed=42):
    rng = np.random.default_rng(seed)

    states      = ['Punjab', 'Maharashtra', 'Karnataka', 'UP', 'Rajasthan', 'MP', 'Gujarat', 'Bihar']
    crop_types  = ['Rice', 'Wheat', 'Bajra', 'Maize', 'Cotton', 'Sugarcane']
    soil_types  = ['Loamy', 'Sandy', 'Clay', 'Black', 'Red']

    state_arr     = rng.choice(states,     n)
    crop_arr      = rng.choice(crop_types, n)
    soil_arr      = rng.choice(soil_types, n)
    year_arr      = rng.integers(2000, 2024, n)
    rainfall_arr  = rng.normal(800, 300, n).clip(100, 2000)
    irrig_arr     = rng.normal(50, 30, n).clip(1, 200)

    # Crop-specific base yields
    crop_base = {'Rice': 3500, 'Wheat': 3200, 'Bajra': 2000,
                 'Maize': 2800, 'Cotton': 1800, 'Sugarcane': 7000}
    soil_mult  = {'Loamy': 1.15, 'Black': 1.10, 'Clay': 1.00, 'Red': 0.90, 'Sandy': 0.80}
    state_mult = {'Punjab': 1.20, 'UP': 1.10, 'Bihar': 0.95, 'Rajasthan': 0.85,
                  'Maharashtra': 1.05, 'Karnataka': 1.00, 'MP': 0.98, 'Gujarat': 1.02}

    yields = np.array([
        crop_base[c] * soil_mult[s] * state_mult[st]
        * (1 + 0.0003 * r) * (1 + 0.002 * i) + rng.normal(0, 200)
        for c, s, st, r, i in zip(crop_arr, soil_arr, state_arr, rainfall_arr, irrig_arr)
    ]).clip(500, 12000)

    return pd.DataFrame({
        'Year': year_arr,
        'State': state_arr,
        'Crop_Type': crop_arr,
        'Rainfall': rainfall_arr,
        'Soil_Type': soil_arr,
        'Irrigation_Area': irrig_arr,
        'Crop_Yield (kg/ha)': yields,
    })


# ─── Model training ───────────────────────────────────────────────────────────
@st.cache_resource
def train_model(df):
    data = df.copy()
    encoders, cat_cols = {}, ['State', 'Crop_Type', 'Soil_Type']
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    scaler = StandardScaler()
    num_cols = ['Rainfall', 'Irrigation_Area']
    data[num_cols] = scaler.fit_transform(data[num_cols])

    X = data.drop('Crop_Yield (kg/ha)', axis=1)
    y = data['Crop_Yield (kg/ha)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200, max_depth=20,
        min_samples_split=5, min_samples_leaf=2,
        max_features='sqrt', random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'r2':   r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'n_train': len(X_train),
        'n_test':  len(X_test),
    }
    return model, encoders, scaler, metrics, X.columns.tolist()


# ─── Prediction helper ────────────────────────────────────────────────────────
def predict(model, encoders, scaler, year, state, crop, rainfall, soil, irrig):
    row = pd.DataFrame([{
        'Year': year,
        'State': encoders['State'].transform([state])[0],
        'Crop_Type': encoders['Crop_Type'].transform([crop])[0],
        'Rainfall': rainfall,
        'Soil_Type': encoders['Soil_Type'].transform([soil])[0],
        'Irrigation_Area': irrig,
    }])
    num_cols = ['Rainfall', 'Irrigation_Area']
    row[num_cols] = scaler.transform(row[num_cols])
    return model.predict(row)[0]


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
df = generate_data()
model, encoders, scaler, metrics, feature_cols = train_model(df)

states     = sorted(df['State'].unique())
crops      = sorted(df['Crop_Type'].unique())
soils      = sorted(df['Soil_Type'].unique())

# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🌾 CropYield AI</h1>
  <h2>Sayan Nurmukhambet, Aidar Amangeldy, Bahyt Madenali</h2>
  <p>Predict crop yield using rainfall, soil type, and irrigation data with Random Forest.</p>
</div>
""", unsafe_allow_html=True)

# ─── Top metrics ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="value">{metrics["r2"]*100:.1f}%</div><div class="label">Model R² Score</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="value">{metrics["rmse"]:.0f}</div><div class="label">RMSE (kg/ha)</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="value">{metrics["n_train"]:,}</div><div class="label">Training samples</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="value">{len(crops)}</div><div class="label">Crop types</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Predict Yield", "📊 Data Insights", "🔬 Model Analysis"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div class="section-header">Input Parameters</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            year    = st.selectbox("📅 Year",   list(range(2000, 2026)), index=22)
            state   = st.selectbox("🗺️ State",  states)
            soil    = st.selectbox("🌱 Soil Type", soils)
        with col_b:
            crop      = st.selectbox("🌾 Crop Type", crops)
            rainfall  = st.slider("🌧️ Rainfall (mm)", 100, 2000, 750, step=10)
            irrigation = st.slider("🚜 Irrigation Area (ha × 1000)", 1, 200, 50)

        predict_btn = st.button("🔮 Predict Crop Yield", use_container_width=True, type="primary")

    with right:
        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            result = predict(model, encoders, scaler, year, state, crop, rainfall, soil, irrigation)

            st.markdown(f"""
            <div class="prediction-box">
                <div style="font-size:0.85rem; color:#999; margin-bottom:8px;">Estimated Crop Yield</div>
                <div class="yield-value">{result:,.0f}</div>
                <div class="unit">kg / hectare</div>
                <div style="font-size:0.85rem; color:#999; margin-top:12px">
                    {crop} · {soil} soil · {rainfall} mm rainfall
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Contextual chart — distribution of yields for this crop
            st.markdown("<br>", unsafe_allow_html=True)
            subset = df[df['Crop_Type'] == crop]['Crop_Yield (kg/ha)']
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.hist(subset, bins=40, color='#ccc', alpha=0.7, edgecolor='#999')
            ax.axvline(result, color='#333', linewidth=2, linestyle='--', label=f'Prediction: {result:,.0f}')
            ax.set_xlabel('Crop Yield (kg/ha)', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.set_title(f'{crop} yield distribution', fontsize=10, fontweight='normal')
            ax.legend(fontsize=8)
            ax.spines[['top','right']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("👈  Fill in the parameters on the left and click **Predict Crop Yield**.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Crop Yield by State & Crop Type</div>', unsafe_allow_html=True)

    grouped = df.groupby(['State', 'Crop_Type'])['Crop_Yield (kg/ha)'].mean().reset_index()
    fig = px.bar(
        grouped, x='State', y='Crop_Yield (kg/ha)', color='Crop_Type',
        barmode='group', template='plotly_white',
        labels={'Crop_Yield (kg/ha)': 'Avg Yield (kg/ha)'},
    )
    fig.update_layout(legend_title_text='Crop Type', height=380)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Rainfall vs Yield</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            df.sample(1000, random_state=1), x='Rainfall', y='Crop_Yield (kg/ha)',
            color='Crop_Type', opacity=0.5, template='plotly_white',
            trendline='ols',
        )
        fig2.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Yield by Soil Type</div>', unsafe_allow_html=True)
        fig3 = px.box(
            df, x='Soil_Type', y='Crop_Yield (kg/ha)',
            color='Soil_Type', template='plotly_white',
        )
        fig3.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Avg Yield Over Years</div>', unsafe_allow_html=True)
    yearly = df.groupby('Year')['Crop_Yield (kg/ha)'].mean().reset_index()
    fig4 = px.line(yearly, x='Year', y='Crop_Yield (kg/ha)',
                   markers=True, template='plotly_white')
    fig4.update_layout(height=300)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-header">Heatmap: Soil × Crop → Avg Yield</div>', unsafe_allow_html=True)
    pivot = df.pivot_table(index='Soil_Type', columns='Crop_Type',
                           values='Crop_Yield (kg/ha)', aggfunc='mean')
    fig5, ax5 = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='gray', ax=ax5, linewidths=0.5, cbar_kws={'label': 'Avg Yield (kg/ha)'})
    ax5.set_title('Average Crop Yield by Soil Type and Crop Type', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig5)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Feature Importances</div>', unsafe_allow_html=True)

    importances = pd.DataFrame({
        'Feature':    feature_cols,
        'Importance': model.feature_importances_,
    }).sort_values('Importance', ascending=True)

    fig_fi = px.bar(importances, x='Importance', y='Feature',
                    orientation='h', template='plotly_white')
    fig_fi.update_layout(height=350)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-header">Model Configuration</div>', unsafe_allow_html=True)
    config = {
        'Algorithm': 'Random Forest Regressor',
        'n_estimators': 200,
        'max_depth': 20,
        'max_features': 'sqrt',
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'Train / Test split': '80% / 20%',
        'R² Score': f"{metrics['r2']*100:.2f}%",
        'RMSE': f"{metrics['rmse']:.2f} kg/ha",
    }
    st.table(pd.DataFrame(list(config.items()), columns=['Parameter', 'Value']))

    st.markdown('<div class="section-header">Actual vs Predicted (test set sample)</div>', unsafe_allow_html=True)

    # Quick re-compute for scatter
    @st.cache_data
    def get_test_predictions():
        d = df.copy()
        cat_cols = ['State', 'Crop_Type', 'Soil_Type']
        for col in cat_cols:
            d[col] = encoders[col].transform(d[col])
        d[['Rainfall','Irrigation_Area']] = scaler.transform(d[['Rainfall','Irrigation_Area']])
        X = d.drop('Crop_Yield (kg/ha)', axis=1)
        y = d['Crop_Yield (kg/ha)']
        _, Xt, _, yt = train_test_split(X, y, test_size=0.2, random_state=42)
        yp = model.predict(Xt)
        return yt.values[:300], yp[:300]

    y_actual, y_pred_vals = get_test_predictions()
    fig_s = px.scatter(
        x=y_actual, y=y_pred_vals,
        labels={'x': 'Actual Yield', 'y': 'Predicted Yield'},
        template='plotly_white', opacity=0.6,
    )
    max_val = max(y_actual.max(), y_pred_vals.max())
    fig_s.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                               mode='lines', line=dict(dash='dash', color='#999'),
                               name='Perfect prediction'))
    fig_s.update_layout(height=380)
    st.plotly_chart(fig_s, use_container_width=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#999; font-size:0.8rem'>"
    "CropYield AI"
    "</div>",
    unsafe_allow_html=True,
)