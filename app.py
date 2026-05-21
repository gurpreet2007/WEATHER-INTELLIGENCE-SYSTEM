import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

df = pd.read_csv("dataset/weather_sample.csv")
# Convert columns
df['RF'] = pd.to_numeric(df['RF'], errors='coerce')

# Target column
df['Rain'] = df['RF'].apply(
    lambda x: 1 if x > 0 else 0
)
numeric_cols = [
    'SLP', 'MSLP', 'DBT', 'WBT',
    'DPT', 'RH', 'VP', 'FFF',
    'VV', 'TC'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
# Create Season column
def get_season(month):

    if month in [12, 1, 2]:
        return 0

    elif month in [3, 4, 5]:
        return 1

    elif month in [6, 7, 8, 9]:
        return 2

    else:
        return 3

df['Season'] = df['MN'].apply(get_season)    

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Weather Intelligence System",
    page_icon="🌦",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/weather_xgboost_model.joblib")
features_final = joblib.load(
    "models/feature.joblib"
)


# # ---------------- TITLE ----------------
# st.title("🌦 Weather Intelligence System")
# st.markdown("### Machine Learning Powered Rainfall Forecasting & Weather Analytics")

# st.write("---")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Rainfall Prediction",
        "Weather Dashboard",
        "Weather Insights",
        "About Project"
    ]
)

# =========================================================
# HOME PAGE
# =========================================================

if page == "Home":
    
    st.title("🌦 Weather Intelligence System")

    st.markdown("""
    ### Machine Learning-Based Rainfall Prediction & Weather Analytics
    """)

    st.write("---")
    st.image("https://cdn.mos.cms.futurecdn.net/YGYuHGH442GU597qspz2YL-1000-80.jpg.webp",width=400)

    st.write("""
    The Weather Intelligence System is a Machine Learning-based application
    developed to predict rainfall using atmospheric and meteorological parameters.

    The system analyzes weather conditions such as temperature, humidity,
    pressure, wind speed, visibility, and seasonal patterns to generate
    rainfall predictions and probability estimates.

    In addition to rainfall prediction, the system also provides interactive
    weather analytics and visualization dashboards for understanding weather trends
    and seasonal rainfall behavior.
    """)

    st.write("---")

    # KPI Cards
    col1, col2, col3 = st.columns(3)

    col1.metric(
        label="Model Accuracy",
        value="73%"
    )

    col2.metric(
        label="ML Model",
        value="XGBoost"
    )

    col3.metric(
        label="Prediction System",
        value="Rainfall Forecast"
    )

    st.write("---")

    st.subheader("🔍 Core Capabilities")

    st.write("""
    • Rainfall Prediction using XGBoost  
    • Rain Probability Estimation  
    • Seasonal Weather Analysis  
    • Interactive Weather Dashboard  
    • Feature Importance Visualization  
    • Atmospheric Data Analysis  
    """)

    st.write("---")

    st.success("""
    This system helps users analyze weather conditions and predict the
    likelihood of rainfall using Machine Learning techniques.
    """)
# =========================================================
# PREDICTION PAGE
# =========================================================
elif page == "Rainfall Prediction":
    
    st.header("🌧 Rainfall Prediction")

    st.write("Enter weather conditions below:")

    col1, col2 = st.columns(2)

    with col1:

        SLP = st.slider(
            "Station Level Pressure (SLP)",
            970.0, 1010.0, 990.0
        )

        MSLP = st.slider(
            "Mean Sea Level Pressure (MSLP)",
            990.0, 1035.0, 1015.0
        )

        DBT = st.slider(
            "Dry Bulb Temperature (DBT)",
            0.0, 45.0, 28.0
        )

        WBT = st.slider(
            "Wet Bulb Temperature (WBT)",
            0.0, 40.0, 24.0
        )

        RH = st.slider(
            "Relative Humidity (RH)",
            0, 100, 85
        )

    with col2:

        FFF = st.slider(
            "Wind Speed (FFF)",
            0.0, 40.0, 12.0
        )

        VV = st.slider(
            "Visibility (VV)",
            0.0, 100.0, 90.0
        )

        Season = st.selectbox(
            "Season",
            [0, 1, 2, 3],
            format_func=lambda x:
            {
                0: "Winter",
                1: "Summer",
                2: "Monsoon",
                3: "Post-Monsoon"
            }[x]
        )

        MN = st.slider(
            "Month",
            1, 12, 7
        )

    # Feature Engineering
    Temp_Humidity = DBT * RH
    Pressure_Diff = MSLP - SLP

    # Prediction
    if st.button("Predict Rainfall"):

        input_data = pd.DataFrame([{
            'SLP': SLP,
            'MSLP': MSLP,
            'DBT': DBT,
            'WBT': WBT,
            'RH': RH,
            'FFF': FFF,
            'VV': VV,
            'Season': Season,
            'MN': MN,
            'Temp_Humidity': Temp_Humidity,
            'Pressure_Diff': Pressure_Diff
        }])

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0][1]

        st.write("---")

        # Probability Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Rain Probability"},
            gauge={
                'axis': {'range': [0, 100]}
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # Prediction Result
        if prediction == 1:
            st.success("🌧 Rain Expected")
        else:
            st.warning("☀ No Rain Expected")

        # Risk Level
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.7:
            risk = "Moderate"
        else:
            risk = "High"

        st.subheader(f"⚠ Weather Risk Level: {risk}")

        # Intelligent Insights
        st.subheader(" Weather Insights")

        if RH > 80:
            st.write("• High humidity increases rainfall chances.")

        if Pressure_Diff > 20:
            st.write("• Atmospheric pressure variation indicates unstable weather.")

        if Season == 2:
            st.write("• Monsoon season strongly supports rainfall occurrence.")

        if FFF > 15:
            st.write("• Strong wind speed may contribute to weather changes.")

# =========================================================
# DASHBOARD PAGE
# =========================================================
elif page == "Weather Dashboard":
    
    st.header("📊 Weather Analytics Dashboard")

    st.markdown("Real-time analytical insights from historical weather patterns.")

    # =========================
    # Monthly Rainfall Trend
    # =========================

    st.subheader("🌧 Monthly Rainfall Trend")

    monthly_rain = df.groupby('MN')['Rain'].mean().reset_index()

    fig1 = px.line(
        monthly_rain,
        x='MN',
        y='Rain',
        markers=True,
        title="Average Monthly Rainfall Probability"
    )

    fig1.update_layout(
        xaxis_title="Month",
        yaxis_title="Rainfall Probability"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # =========================
    # Temperature Distribution
    # =========================

    st.subheader("🌡 Temperature Distribution")

    fig2 = px.histogram(
        df,
        x='DBT',
        nbins=30,
        title="Dry Bulb Temperature Distribution"
    )

    fig2.update_layout(
        xaxis_title="Temperature",
        yaxis_title="Frequency"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # Humidity vs Rainfall
    # =========================

    st.subheader("💧 Humidity vs Rainfall")

    fig3 = px.box(
        df,
        x='Rain',
        y='RH',
        title="Humidity Distribution for Rainy vs Non-Rainy Days"
    )

    fig3.update_layout(
        xaxis_title="Rain",
        yaxis_title="Relative Humidity"
    )

    st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # Correlation Heatmap
    # =========================

    st.subheader("🔥 Feature Correlation Heatmap")

    correlation = df[
        ['SLP', 'MSLP', 'DBT', 'WBT',
         'DPT', 'RH', 'VP', 'FFF',
         'VV', 'TC', 'Rain']
    ].corr()

    fig4 = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix"
    )

    st.plotly_chart(fig4, use_container_width=True)

    # =========================
    # Feature Importance
    # =========================

    st.subheader("⭐ XGBoost Feature Importance")

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        'Feature': features_final,
        'Importance': importance
    })

    feature_importance = feature_importance.sort_values(
        by='Importance',
        ascending=True
    )

    fig5 = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Most Influential Weather Features"
    )

    st.plotly_chart(fig5, use_container_width=True)

    # =========================
    # Seasonal Rainfall Analysis
    # =========================

    st.subheader("🌦 Seasonal Rainfall Analysis")

    seasonal_rain = df.groupby('Season')['Rain'].mean().reset_index()

    season_map = {
        0: "Winter",
        1: "Summer",
        2: "Monsoon",
        3: "Post-Monsoon"
    }

    seasonal_rain['Season_Name'] = seasonal_rain['Season'].map(season_map)

    fig6 = px.bar(
        seasonal_rain,
        x='Season_Name',
        y='Rain',
        title="Rainfall Probability by Season"
    )

    fig6.update_layout(
        xaxis_title="Season",
        yaxis_title="Rainfall Probability"
    )

    st.plotly_chart(fig6, use_container_width=True)

# =========================================================
# INSIGHTS PAGE
# =========================================================

elif page == "Weather Insights":

    st.header("Weather Insights")

    st.info("""
    Monsoon season historically shows the highest rainfall probability.
    """)

    st.info("""
    Relative Humidity above 80% strongly correlates with rainfall occurrence.
    """)

    st.info("""
    Lower atmospheric pressure generally increases rainfall likelihood.
    """)

    st.info("""Changes in wind speed and reduced visibility are commonly associated with unstable weather conditions.""")
    
    

# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "About Project":
    
    st.header("About Weather Intelligence System")

    st.write("""
    The Weather Intelligence System is a Machine Learning-based web application
    designed to predict the probability of rainfall using atmospheric and
    meteorological parameters.

    This project combines data analysis, feature engineering, machine learning,
    and interactive visualization to provide weather insights in a user-friendly dashboard.
    """)

    st.subheader("🎯 Project Objective")

    st.write("""
    The main objective of this project is to assist in early rainfall prediction
    using historical weather observations such as temperature, humidity,
    pressure, wind speed, and visibility.

    Accurate rainfall prediction can help in:
    - Agriculture planning
    - Disaster preparedness
    - Water resource management
    - Smart city monitoring
    - Weather analysis systems
    """)

    st.subheader("🤖 Machine Learning Workflow")

    st.write("""
    The system was trained using multiple machine learning algorithms including:
    - Logistic Regression
    - Random Forest
    - Support Vector Machine (SVM)
    - Stacking Classifier
    - XGBoost

    After experimentation and hyperparameter tuning,
    XGBoost delivered the best performance.
    """)

    st.subheader("📊 Model Performance")

    st.write("""
    Final Model: XGBoost Classifier

    Performance Metrics:
    - Accuracy: ~73%
    - Strong rainfall detection capability
    - Optimized using feature engineering and parameter tuning

    Engineered Features:
    - Season
    - Temperature-Humidity Interaction
    - Pressure Difference
    """)

    st.subheader("🛠 Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Pandas & NumPy
    - Scikit-learn
    - XGBoost
    - Plotly
    - Matplotlib & Seaborn
    """)

    st.subheader("🚀 Future Improvements")

    st.write("""
    Future enhancements may include:
    - Live weather API integration
    - Multi-day rainfall forecasting
    - Temperature prediction
    - Wind speed forecasting
    - Severe weather alerts
    - Deployment on cloud platforms
    """)
    
    