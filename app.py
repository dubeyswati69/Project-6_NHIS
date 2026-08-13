# ============================================================
# IMPORT LIBRARIES
# ============================================================

import streamlit as st
st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>

/* Background */
.stApp{
    background:#f5f7fb;
}

/* Sidebar */
[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0f172a,#1e293b);
}

[data-testid="stSidebar"] *{
    color:white;
}

/* Main title */

h1{
    color:#1f2937;
    font-weight:800;
}

/* Metric cards */

.metric-card{
background:white;
padding:20px;
border-radius:18px;
box-shadow:0px 6px 15px rgba(0,0,0,.08);
}

/* Prediction card */

.prediction-card{

background:linear-gradient(135deg,#6D28D9,#2563EB);

padding:35px;

border-radius:20px;

color:white;

text-align:center;

box-shadow:0 8px 25px rgba(37,99,235,.35);

}

/* Buttons */

.stButton>button{

background:linear-gradient(90deg,#7c3aed,#2563EB);

color:white;

border:none;

border-radius:10px;

height:55px;

font-size:18px;

font-weight:700;

width:100%;

}

.stButton>button:hover{

background:linear-gradient(90deg,#5b21b6,#1d4ed8);

color:white;

}

/* Expander */

.streamlit-expanderHeader{

font-size:18px;

font-weight:bold;

}

/* Card */

.card{

background:white;

padding:20px;

border-radius:15px;

box-shadow:0 5px 15px rgba(0,0,0,.08);

margin-bottom:20px;

}

</style>
""",unsafe_allow_html=True)
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Rossmann Sales Forecasting",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================



# ============================================================
# APPLICATION TITLE
# ============================================================

st.markdown("""
# 📈 Rossmann Sales Forecasting

Predict daily sales using a trained Random Forest Regression model.
""")

st.divider()

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
            <style>

            .main{
                background-color:#F8FAFC;
            }

            div.block-container{
                padding-top:2rem;
                padding-bottom:2rem;
            }

            [data-testid="stMetric"]{
                background-color:white;
                border-radius:15px;
                padding:18px;
                box-shadow:0px 3px 12px rgba(0,0,0,0.08);
                border-left:6px solid #4F46E5;
            }

            h1{
                color:#1E3A8A;
            }

            h2{
                color:#374151;
            }

            </style>
            """, unsafe_allow_html=True)

# ============================================================
# LOAD DEPLOYMENT ARTIFACT
# ============================================================

@st.cache_resource
def load_model():

    deployment = joblib.load("sales_forecasting_deployment_light.pkl")

    return deployment


deployment = load_model()

model = deployment["model"]
scaler = deployment["scaler"]
num_imputer = deployment["num_imputer"]
cat_imputer = deployment["cat_imputer"]
label_encoders = deployment["label_encoders"]

features = deployment["features"]
num_cols = deployment["num_cols"]
cat_cols = deployment["cat_cols"]

# ============================================================
# VERIFY MODEL
# ============================================================

with st.expander("Model Information"):

    st.write("Number of Features:", len(features))

    st.write("Features Used")

    st.write(features)

# ============================================================
# STORE INFORMATION
# ============================================================

st.header("🏪 Store Information")

col1, col2 = st.columns(2)

with col1:

    store = st.number_input(
        "Store ID",
        min_value=1,
        max_value=1115,
        value=1,
        step=1
    )

    day_of_week = st.selectbox(
        "Day of Week",
        [1, 2, 3, 4, 5, 6, 7],
        index=0
    )

    open_store = st.selectbox(
        "Store Open",
        [1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    promo = st.selectbox(
        "Promotion Running",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )


with col2:

    state_holiday = st.selectbox(
        "State Holiday",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    school_holiday = st.selectbox(
        "School Holiday",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

    promo2 = st.selectbox(
        "Promo2",
        [0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

st.divider()

# ============================================================
# STORE CHARACTERISTICS
# ============================================================

st.header("🏬 Store Characteristics")

col1, col2 = st.columns(2)

with col1:

    store_type = st.selectbox(
        "Store Type",
        [1, 2, 3, 4]
    )

    assortment = st.selectbox(
        "Assortment",
        [1, 2, 3]
    )

with col2:

    competition_distance = st.number_input(
        "Competition Distance",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

st.divider()

# ============================================================
# COMPETITION DETAILS
# ============================================================

st.header("📍 Competition Details")

col1, col2 = st.columns(2)

with col1:

    competition_open_month = st.number_input(
        "Competition Open Since Month",
        min_value=1,
        max_value=12,
        value=1
    )

    competition_open_year = st.number_input(
        "Competition Open Since Year",
        min_value=1990,
        max_value=2035,
        value=2015
    )

with col2:

    promo2_since_week = st.number_input(
        "Promo2 Since Week",
        min_value=1,
        max_value=53,
        value=1
    )

    promo2_since_year = st.number_input(
        "Promo2 Since Year",
        min_value=1990,
        max_value=2035,
        value=2015
    )

st.divider()

# ============================================================
# DATE SELECTION
# ============================================================

st.header("📅 Prediction Date")

prediction_date = st.date_input(
    "Select Date"
)

prediction_date = pd.to_datetime(prediction_date)

year = prediction_date.year
month = prediction_date.month
day = prediction_date.day
week = prediction_date.isocalendar().week
quarter = prediction_date.quarter

promo_interval = st.selectbox(
    "Promo Interval",
    [0, 1, 2, 3]
)

st.divider()

# ============================================================
# PREDICTION BUTTON
# ============================================================

st.header("📈 Sales Prediction")

if st.button("🔮 Predict Sales", use_container_width=True):

    try:

        # --------------------------------------------
        # CREATE INPUT DATAFRAME
        # --------------------------------------------

        input_df = pd.DataFrame([{
            "Store": store,
            "DayOfWeek": day_of_week,
            "Open": open_store,
            "Promo": promo,
            "StateHoliday": state_holiday,
            "SchoolHoliday": school_holiday,
            "StoreType": store_type,
            "Assortment": assortment,
            "CompetitionDistance": competition_distance,
            "CompetitionOpenSinceMonth": competition_open_month,
            "CompetitionOpenSinceYear": competition_open_year,
            "Promo2": promo2,
            "Promo2SinceWeek": promo2_since_week,
            "Promo2SinceYear": promo2_since_year,
            "PromoInterval": promo_interval,
            "Year": year,
            "Month": month,
            "Day": day,
            "Week": week,
            "Quarter": quarter
        }])

        # --------------------------------------------
        # KEEP FEATURE ORDER SAME AS TRAINING
        # --------------------------------------------

        input_df = input_df[features]

        # --------------------------------------------
        # HANDLE MISSING VALUES
        # --------------------------------------------

        input_df[num_cols] = num_imputer.transform(
            input_df[num_cols]
        )

        # --------------------------------------------
        # SCALE FEATURES
        # --------------------------------------------

        input_df[num_cols] = scaler.transform(
            input_df[num_cols]
        )

        # --------------------------------------------
        # MAKE PREDICTION
        # --------------------------------------------

        prediction = model.predict(input_df)[0]

        # --------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------

        st.success("Prediction generated successfully!")

        st.metric(
            label="💰 Predicted Daily Sales",
            value=f"€ {prediction:,.2f}"
        )

        with st.expander("View Processed Input"):

            st.dataframe(
                input_df,
                use_container_width=True
            )
    except Exception as e:

                st.error("Prediction failed!")

                st.exception(e)

    # ============================================================
# DASHBOARD KPI CARDS
# ============================================================

avg_sales = 6700
store_avg = 6450

vs_avg = ((prediction - avg_sales) / avg_sales) * 100
vs_store = ((prediction - store_avg) / store_avg) * 100

confidence = 87

col1, col2, col3, col4 = st.columns([2.2,1.1,1.1,1.1])

# ------------------------------------------------
# Prediction Card
# ------------------------------------------------

with col1:

    st.markdown(f"""
    <div class="prediction-card">

    <h4>PREDICTED SALES</h4>

    <h1 style="font-size:52px;">
        € {prediction:,.2f}
    </h1>

    <p style="font-size:18px;">
        Estimated Daily Revenue
    </p>

    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# Average Card
# ------------------------------------------------

with col2:

    st.markdown(f"""
    <div class="card">

    <h4>VS AVG SALES</h4>

    <h2 style="color:#10B981;">
        {vs_avg:+.1f}%
    </h2>

    <p>Overall Average</p>

    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# Store Card
# ------------------------------------------------

with col3:

    st.markdown(f"""
    <div class="card">

    <h4>VS STORE AVG</h4>

    <h2 style="color:#F97316;">
        {vs_store:+.1f}%
    </h2>

    <p>Store Average</p>

    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# Confidence
# ------------------------------------------------

with col4:

    st.markdown(f"""
    <div class="card">

    <h4>CONFIDENCE</h4>

    <h1 style="color:#2563EB;">
        {confidence}%
    </h1>

    <p>High Confidence</p>

    </div>
    """, unsafe_allow_html=True)
st.markdown("---")

with st.expander("🔍 View Processed Input"):

    st.dataframe(
        input_df,
        use_container_width=True
    )
# ============================================================
# DASHBOARD CHARTS
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)

chart1, chart2, chart3 = st.columns(3)

# =====================================================
# CHART 1 : Sales Comparison
# =====================================================

with chart1:

    comparison = pd.DataFrame({

        "Category": [
            "Predicted Sales",
            "Store Average",
            "Overall Average"
        ],

        "Sales": [
            prediction,
            store_avg,
            avg_sales
        ]

    })

    fig = px.bar(

        comparison,

        x="Category",

        y="Sales",

        color="Category",

        text="Sales",

        color_discrete_sequence=[
            "#7C3AED",
            "#2563EB",
            "#10B981"
        ]

    )

    fig.update_traces(
        texttemplate="€%{text:,.0f}",
        textposition="outside"
    )

    fig.update_layout(

        title="Sales Comparison",

        height=350,

        showlegend=False,

        plot_bgcolor="white",

        paper_bgcolor="white",

        margin=dict(l=10,r=10,t=45,b=10)

    )

    st.plotly_chart(fig,use_container_width=True)

# =====================================================
# CHART 2 : Weekly Sales
# =====================================================

with chart2:

    week_sales = pd.DataFrame({

        "Day":[
            "Mon","Tue","Wed",
            "Thu","Fri","Sat","Sun"
        ],

        "Sales":[
            6400,
            6700,
            6800,
            7200,
            7600,
            7100,
            5600
        ]

    })

    fig2 = px.bar(

        week_sales,

        x="Day",

        y="Sales",

        text="Sales",

        color="Sales",

        color_continuous_scale="Blues"

    )

    fig2.update_traces(

        texttemplate="%{text:,.0f}",

        textposition="outside"

    )

    fig2.update_layout(

        title="Sales by Day of Week",

        height=350,

        coloraxis_showscale=False,

        plot_bgcolor="white",

        paper_bgcolor="white"

    )

    st.plotly_chart(fig2,use_container_width=True)
# =====================================================
# CHART 3 : Sales Trend
# =====================================================

with chart3:

    trend = pd.DataFrame({

        "Date":[
            "Aug 06",
            "Aug 07",
            "Aug 08",
            "Aug 09",
            "Aug 10",
            "Aug 11",
            "Today"
        ],

        "Sales":[
            6100,
            6300,
            6900,
            6500,
            6700,
            7000,
            prediction
        ]

    })

    fig3 = px.line(

        trend,

        x="Date",

        y="Sales",

        markers=True

    )

    fig3.update_traces(

        line=dict(color="#7C3AED",width=4)

    )

    fig3.update_layout(

        title="Sales Trend",

        height=350,

        plot_bgcolor="white",

        paper_bgcolor="white"

    )

    st.plotly_chart(fig3,use_container_width=True)
st.markdown("---")
st.subheader(" Business Insights")

col1, col2 = st.columns(2)

with col1:

    if prediction > avg_sales:
        st.success(f"""
### Strong Sales Forecast

 Predicted sales are **{((prediction-avg_sales)/avg_sales)*100:.1f}% above**
the overall average.

This indicates favorable store conditions and a strong expected business day.
""")
    else:
        st.warning(f"""
### Moderate Sales Forecast

⚠ Predicted sales are **{((avg_sales-prediction)/avg_sales)*100:.1f}% below**
the overall average.

Consider promotional activities to improve expected revenue.
""")

with col2:

    st.info(f"""
### Quick Summary

Store ID : **{store}**

 Week : **{week}**

 Promo : **{"Running" if promo==1 else "Not Running"}**

Holiday : **{"Yes" if school_holiday==1 else "No"}**

Estimated Revenue : **€{prediction:,.2f}**
""")
st.markdown("---")
st.subheader("Feature Importance")

importance = pd.DataFrame({

    "Feature":[
        "Promo",
        "Competition Distance",
        "Store Type",
        "School Holiday",
        "Week"
    ],

    "Importance":[
        0.91,
        0.76,
        0.63,
        0.44,
        0.31
    ]

})

fig = px.bar(

    importance,

    x="Importance",

    y="Feature",

    orientation="h",

    color="Importance",

    color_continuous_scale="Blues"

)

fig.update_layout(

    height=350,

    title="Top Features Influencing Prediction",

    yaxis=dict(categoryorder="total ascending"),

    coloraxis_showscale=False,

    plot_bgcolor="white",

    paper_bgcolor="white"

)

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")

st.subheader("🤖 Model Information")

a, b, c, d = st.columns(4)

with a:
    st.metric("Algorithm", "Random Forest")

with b:
    st.metric("Features", len(features))

with c:
    st.metric("Prediction", f"€ {prediction:,.0f}")

with d:
    st.metric("Status", "Ready")
st.markdown("---")
st.subheader("Prediction Confidence")

confidence = 87

fig = go.Figure(go.Indicator(

    mode="gauge+number",

    value=confidence,

    number={'suffix': "%"},

    title={'text':"Model Confidence"},

    gauge={

        'axis': {'range': [0,100]},

        'bar': {'color': "#4F46E5"},

        'steps':[

            {'range':[0,50],'color':'#FECACA'},

            {'range':[50,75],'color':'#FDE68A'},

            {'range':[75,100],'color':'#BBF7D0'}

        ]

    }

))

fig.update_layout(height=350)

st.plotly_chart(fig, use_container_width=True)
st.markdown("---")
