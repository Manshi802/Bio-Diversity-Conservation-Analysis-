"""
app.py
------
AI Based Biodiversity Conservation Analytics
Streamlit Web Dashboard for Species Risk Assessment and Conservation Insights.
"""

import os
import sys

# Ensure project root is accessible
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import MarkerCluster
import streamlit.components.v1 as components

# Import local modules
from src.preprocess import load_data, get_data_summary, CATEGORY_ORDER, STATUS_ORDER
from src.model import load_or_train_model, predict_conservation_risk
from src.insights import generate_conservation_recommendations, get_top_at_risk_regions

# ---------------------------------------------------------
# Page Configuration and Global Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Biodiversity Conservation Analytics",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean aesthetics, metric cards, and alert badges
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #40916c;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 5px solid #2d6a4f;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 1.8rem;
        color: #1b4332;
    }
    .metric-card p {
        margin: 0;
        font-size: 0.9rem;
        color: #555;
    }
    .badge-critical {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-endangered {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-vulnerable {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-least-concern {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Standard IUCN Color Palette
IUCN_COLORS = {
    "Least Concern": "#2ecc71",          # Green
    "Vulnerable": "#f1c40f",             # Yellow
    "Endangered": "#e67e22",             # Orange
    "Critically Endangered": "#e74c3c"   # Red
}

# ---------------------------------------------------------
# Cached Data and Model Loaders
# ---------------------------------------------------------
@st.cache_data
def get_cached_dataset():
    """Loads and caches species dataset."""
    return load_data("data/species_data.csv")

@st.cache_resource
def get_cached_model():
    """Loads or trains the Random Forest classifier and caches in memory."""
    return load_or_train_model("models/model.pkl")

# Helper function for Plotly scatter map across different Plotly versions
def render_plotly_map(df_map):
    """Creates a Plotly map compatible with Plotly 5, 6, and 7."""
    hover_data = {
        "species_name": True,
        "category": True,
        "region": True,
        "population_count": ":,",
        "habitat_loss_percent": ":.1f%",
        "human_activity_index": ":.1f",
        "latitude": False,
        "longitude": False
    }
    
    if hasattr(px, "scatter_map"):
        fig = px.scatter_map(
            df_map,
            lat="latitude",
            lon="longitude",
            color="conservation_status",
            color_discrete_map=IUCN_COLORS,
            hover_name="species_name",
            hover_data=hover_data,
            zoom=1.3,
            height=600,
            map_style="open-street-map"
        )
    else:
        fig = px.scatter_mapbox(
            df_map,
            lat="latitude",
            lon="longitude",
            color="conservation_status",
            color_discrete_map=IUCN_COLORS,
            hover_name="species_name",
            hover_data=hover_data,
            zoom=1.3,
            height=600,
            mapbox_style="open-street-map"
        )
    
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            title_text="Conservation Status",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

# ---------------------------------------------------------
# Application Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1542601906990-b4d3fb778b09?w=400&q=80", caption="Biodiversity AI Analytics")
st.sidebar.title("🌿 Navigation")

menu_options = [
    "📊 Data Overview",
    "📈 Exploratory Data Analysis (EDA)",
    "🗺️ Geospatial Species Map",
    "🤖 AI Risk Prediction",
    "💡 Conservation Insights & Actions"
]

selected_page = st.sidebar.radio("Select Dashboard Section:", menu_options)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Project Details:**
- **Topic:** AI Based Biodiversity Conservation Analytics
- **Model:** Random Forest Classifier
- **Scope:** Species Vulnerability & Regional Risk
""")

# Load Dataset
try:
    df = get_cached_dataset()
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.info("Tip: Run `python src/data_generator.py` in your terminal to create the dataset.")
    st.stop()

# =========================================================
# PAGE 1: DATA OVERVIEW
# =========================================================
if selected_page == "📊 Data Overview":
    st.markdown('<div class="main-title">📊 Species Dataset Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explore global biodiversity records, data integrity, and key environmental indicators.</div>', unsafe_allow_html=True)

    summary = get_data_summary(df)

    # Top KPI metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{summary['total_records']:,}</h3>
            <p>Total Observations</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{summary['total_species']}</h3>
            <p>Distinct Species</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{summary['total_regions']}</h3>
            <p>Hotspot Regions</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(summary['categories'])}</h3>
            <p>Taxonomic Categories</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Data Filter and Preview
    st.subheader("🔍 Dataset Preview & Search")
    filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 2])
    
    with filter_col1:
        cat_filter = st.selectbox("Filter Category:", ["All"] + list(df["category"].unique()))
    with filter_col2:
        status_filter = st.selectbox("Filter Conservation Status:", ["All"] + STATUS_ORDER)
    with filter_col3:
        search_query = st.text_input("Search Species or Region:", placeholder="e.g. Bengal Tiger, Western Ghats...")

    filtered_preview = df.copy()
    if cat_filter != "All":
        filtered_preview = filtered_preview[filtered_preview["category"] == cat_filter]
    if status_filter != "All":
        filtered_preview = filtered_preview[filtered_preview["conservation_status"] == status_filter]
    if search_query:
        query = search_query.strip().lower()
        filtered_preview = filtered_preview[
            filtered_preview["species_name"].str.lower().str.contains(query) |
            filtered_preview["region"].str.lower().str.contains(query)
        ]

    st.write(f"Showing **{len(filtered_preview)}** matching records:")
    st.dataframe(filtered_preview.head(50), use_container_width=True, height=350)

    # Data Quality & Missing Value Check
    st.markdown("---")
    st.subheader("🛡️ Data Completeness & Missing Value Audit")
    
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(df[col].dtype) for col in df.columns],
        "Missing Values Count": [df[col].isnull().sum() for col in df.columns],
        "Missing Percentage (%)": [(df[col].isnull().sum() / len(df)) * 100 for col in df.columns]
    })

    col_m1, col_m2 = st.columns([2, 1])
    with col_m1:
        st.dataframe(missing_df, use_container_width=True)
    with col_m2:
        total_missing = df.isnull().sum().sum()
        if total_missing == 0:
            st.success("✅ **Zero Missing Values Found!** The dataset is complete, consistent, and ready for statistical modeling.")
        else:
            st.warning(f"⚠️ {total_missing} missing values detected.")

    # Descriptive Statistics
    st.markdown("---")
    st.subheader("📈 Statistical Summary of Numerical Features")
    st.dataframe(df.describe().round(2), use_container_width=True)


# =========================================================
# PAGE 2: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
elif selected_page == "📈 Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-title">📈 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Visualizing species distribution, population trends, habitat degradation, and feature correlations.</div>', unsafe_allow_html=True)

    # Row 1: Category Count & Conservation Status Distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Species Count by Category")
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        
        fig1 = px.bar(
            cat_counts,
            x="Category",
            y="Count",
            color="Category",
            text="Count",
            color_discrete_sequence=px.colors.qualitative.Prism,
            title="Distribution across Taxonomic Groups"
        )
        fig1.update_layout(showlegend=False, xaxis_title="Taxonomic Class", yaxis_title="Number of Records")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("2. Conservation Status Distribution")
        status_counts = df["conservation_status"].value_counts().reindex(STATUS_ORDER).reset_index()
        status_counts.columns = ["Conservation Status", "Count"]

        fig2 = px.pie(
            status_counts,
            names="Conservation Status",
            values="Count",
            color="Conservation Status",
            color_discrete_map=IUCN_COLORS,
            hole=0.45,
            title="Proportion of IUCN Threat Levels"
        )
        fig2.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Row 2: Population Trend Over Years & Habitat Loss vs Population Scatter
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3. Median Population Trend Over Years (2015-2024)")
        trend_df = df.groupby(["year", "category"])["population_count"].median().reset_index()
        fig3 = px.line(
            trend_df,
            x="year",
            y="population_count",
            color="category",
            markers=True,
            title="Temporal Population Shift by Category"
        )
        fig3.update_layout(xaxis_title="Year of Survey", yaxis_title="Median Population Count")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.subheader("4. Habitat Loss (%) vs. Population Count")
        fig4 = px.scatter(
            df,
            x="habitat_loss_percent",
            y="population_count",
            color="conservation_status",
            color_discrete_map=IUCN_COLORS,
            hover_name="species_name",
            log_y=True,
            opacity=0.75,
            title="Impact of Habitat Destruction on Population (Log Scale)"
        )
        fig4.update_layout(
            xaxis_title="Habitat Loss (%)",
            yaxis_title="Population Count (Logarithmic)",
            legend_title="IUCN Status"
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # Row 3: Correlation Heatmap
    st.subheader("5. Correlation Heatmap of Ecological Indicators")
    numeric_cols = ["population_count", "habitat_loss_percent", "avg_temperature", "human_activity_index"]
    corr_matrix = df[numeric_cols].corr()

    # Rename for readability
    clean_labels = ["Population Count", "Habitat Loss (%)", "Avg Temperature (°C)", "Human Activity Index"]
    corr_matrix.columns = clean_labels
    corr_matrix.index = clean_labels

    fig5 = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        title="Pearson Correlation Matrix (Ecological & Anthropogenic Variables)"
    )
    st.plotly_chart(fig5, use_container_width=True)


# =========================================================
# PAGE 3: GEOSPATIAL SPECIES MAP
# =========================================================
elif selected_page == "🗺️ Geospatial Species Map":
    st.markdown('<div class="main-title">🗺️ Geospatial Biodiversity Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Interactive geographic mapping of species sightings and regional vulnerability levels.</div>', unsafe_allow_html=True)

    # Map Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_categories = st.multiselect(
            "Select Categories:",
            options=CATEGORY_ORDER,
            default=CATEGORY_ORDER
        )
    with col_f2:
        all_regions = sorted(df["region"].unique())
        selected_regions = st.multiselect(
            "Select Regions:",
            options=all_regions,
            default=all_regions
        )
    with col_f3:
        selected_statuses = st.multiselect(
            "Select Conservation Statuses:",
            options=STATUS_ORDER,
            default=STATUS_ORDER
        )

    # Filter data
    map_df = df[
        (df["category"].isin(selected_categories)) &
        (df["region"].isin(selected_regions)) &
        (df["conservation_status"].isin(selected_statuses))
    ]

    # Map rendering options
    map_col1, map_col2 = st.columns([3, 1])
    with map_col1:
        st.write(f"Displaying **{len(map_df)}** species observations on the map.")
    with map_col2:
        map_view_type = st.radio("Map Engine:", ["Plotly Map", "Folium Marker Clusters"], horizontal=True)

    if len(map_df) == 0:
        st.warning("⚠️ No species records match your selected filter criteria. Please broaden your selection above.")
    else:
        if map_view_type == "Plotly Map":
            fig_map = render_plotly_map(map_df)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            # Folium Map with Marker Clusters
            f_map = folium.Map(location=[15.0, 30.0], zoom_start=2, tiles="CartoDB positron")
            marker_cluster = MarkerCluster().add_to(f_map)

            # Limit folium rendering to maximum 500 points for browser performance
            folium_sample = map_df.sample(min(len(map_df), 400), random_state=42)

            for _, row in folium_sample.iterrows():
                status = row["conservation_status"]
                color = "red" if status == "Critically Endangered" else ("orange" if status == "Endangered" else ("beige" if status == "Vulnerable" else "green"))
                
                popup_html = f"""
                <div style="font-family: sans-serif; width: 180px;">
                    <h4 style="margin: 0 0 5px 0; color: #1b4332;">{row['species_name']}</h4>
                    <b>Category:</b> {row['category']}<br>
                    <b>Status:</b> {row['conservation_status']}<br>
                    <b>Population:</b> {row['population_count']:,}<br>
                    <b>Habitat Loss:</b> {row['habitat_loss_percent']}%<br>
                    <b>Region:</b> {row['region']}
                </div>
                """
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=6,
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=f"{row['species_name']} ({row['conservation_status']})",
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8
                ).add_to(marker_cluster)

            components.html(f_map._repr_html_(), height=550)


# =========================================================
# PAGE 4: ML PREDICTION
# =========================================================
elif selected_page == "🤖 AI Risk Prediction":
    st.markdown('<div class="main-title">🤖 AI Species Risk Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Train and evaluate a Random Forest Classifier to assess species extinction risk.</div>', unsafe_allow_html=True)

    # Load trained model bundle
    try:
        model_bundle = get_cached_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

    # Section 1: Model Performance Metrics
    st.subheader("📊 Model Performance & Diagnostics")
    m_col1, m_col2, m_col3 = st.columns([1, 1.5, 1.5])

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{model_bundle['accuracy']}%</h3>
            <p>Overall Test Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        **Model Details:**
        - Algorithm: **Random Forest**
        - Trees: **100 Estimators**
        - Split: **80% Train / 20% Test**
        - Stratified Multiclass Target
        """)
        
        # Retrain button
        if st.button("🔄 Retrain Model"):
            with st.spinner("Retraining Random Forest model..."):
                model_bundle = load_or_train_model(force_retrain=True)
                st.cache_resource.clear()
                st.success("Model retrained successfully!")
                st.rerun()

    with m_col2:
        st.markdown("**Confusion Matrix (Actual vs. Predicted)**")
        cm = model_bundle["confusion_matrix"]
        classes = model_bundle["classes"]
        
        cm_fig = px.imshow(
            cm,
            x=classes,
            y=classes,
            text_auto=True,
            color_continuous_scale="Greens",
            labels=dict(x="Predicted Class", y="Actual Class")
        )
        cm_fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(cm_fig, use_container_width=True)

    with m_col3:
        st.markdown("**Feature Importance Ranking**")
        feat_df = model_bundle["feature_importances"]
        feat_fig = px.bar(
            feat_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        feat_fig.update_layout(yaxis=dict(autorange="reversed"), height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(feat_fig, use_container_width=True)

    # Classification Report
    with st.expander("📋 Detailed Classification Report (Precision, Recall, F1-Score)"):
        st.dataframe(model_bundle["classification_report_df"].round(3), use_container_width=True)

    st.markdown("---")

    # Section 2: Interactive Prediction Form
    st.subheader("🔮 Predict Conservation Status for a Species")
    st.write("Enter environmental and demographic variables to estimate the species conservation risk level:")

    with st.form("risk_prediction_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            input_category = st.selectbox("Taxonomic Category:", CATEGORY_ORDER)
            input_population = st.number_input(
                "Estimated Population Count:",
                min_value=10,
                max_value=100000,
                value=1200,
                step=50,
                help="Total known mature individuals of the species."
            )
            input_habitat_loss = st.slider(
                "Habitat Loss (%):",
                min_value=0.0,
                max_value=100.0,
                value=45.0,
                step=0.5,
                help="Percentage of natural habitat degraded or fragmented."
            )

        with form_col2:
            input_temp = st.slider(
                "Average Temperature (°C):",
                min_value=-15.0,
                max_value=45.0,
                value=24.5,
                step=0.5,
                help="Mean regional temperature in degrees Celsius."
            )
            input_human_activity = st.slider(
                "Human Activity Index (1.0 to 10.0):",
                min_value=1.0,
                max_value=10.0,
                value=6.0,
                step=0.1,
                help="Anthropogenic disturbance scale (urbanization, agriculture, roads)."
            )
            input_protected = st.radio(
                "Located in a Protected Area (Sanctuary / National Park)?",
                ["Yes", "No"],
                horizontal=True
            )

        submit_btn = st.form_submit_button("🚀 Predict Conservation Risk Level", use_container_width=True)

    if submit_btn:
        predicted_status, prob_dict = predict_conservation_risk(
            model_bundle=model_bundle,
            category=input_category,
            population_count=input_population,
            habitat_loss_percent=input_habitat_loss,
            avg_temperature=input_temp,
            human_activity_index=input_human_activity,
            protected_area=input_protected
        )

        st.markdown("### 🎯 Prediction Results")
        
        # Color badge depending on status
        badge_style = {
            "Critically Endangered": "badge-critical",
            "Endangered": "badge-endangered",
            "Vulnerable": "badge-vulnerable",
            "Least Concern": "badge-least-concern"
        }.get(predicted_status, "badge-least-concern")

        res_col1, res_col2 = st.columns([1.2, 2])
        with res_col1:
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #f1f5f9; border: 1px solid #cbd5e1; text-align: center;">
                <p style="margin: 0; font-size: 1rem; color: #475569;">Predicted IUCN Category</p>
                <h2 style="margin: 10px 0;"><span class="{badge_style}">{predicted_status}</span></h2>
                <p style="margin: 0; font-size: 0.9rem; color: #64748b;">Confidence: <b>{prob_dict[predicted_status]}%</b></p>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown("**Confidence Probability Breakdown:**")
            prob_df = pd.DataFrame(list(prob_dict.items()), columns=["Status", "Probability (%)"])
            prob_fig = px.bar(
                prob_df,
                x="Probability (%)",
                y="Status",
                orientation="h",
                color="Status",
                color_discrete_map=IUCN_COLORS,
                text="Probability (%)"
            )
            prob_fig.update_layout(height=180, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(prob_fig, use_container_width=True)

        # Quick recommendations
        st.markdown("---")
        st.subheader("💡 Immediate Action Recommendations")
        recs = generate_conservation_recommendations(
            habitat_loss_percent=input_habitat_loss,
            population_count=input_population,
            human_activity_index=input_human_activity,
            protected_area=input_protected,
            avg_temperature=input_temp,
            predicted_status=predicted_status
        )

        for rec in recs:
            if rec["level"] == "CRITICAL":
                st.error(f"{rec['icon']} [{rec['domain']}] {rec['action']}")
            elif rec["level"] == "WARNING":
                st.warning(f"{rec['icon']} [{rec['domain']}] {rec['action']}")
            else:
                st.info(f"{rec['icon']} [{rec['domain']}] {rec['action']}")


# =========================================================
# PAGE 5: CONSERVATION INSIGHTS & ACTIONS
# =========================================================
elif selected_page == "💡 Conservation Insights & Actions":
    st.markdown('<div class="main-title">💡 Conservation Insights & Action Plans</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Automated rule-based policy recommendations and regional priority vulnerability assessments.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📋 Species Action Simulator", "🌍 Top 10 At-Risk Regions"])

    with tab1:
        st.subheader("Automated Species Action Generator")
        st.write("Select a species from our database or simulate environmental conditions to generate clear policy guidelines:")

        species_selection = st.selectbox(
            "Select an existing species from records:",
            ["(Custom Scenario)"] + sorted(df["species_name"].unique())
        )

        if species_selection != "(Custom Scenario)":
            sample_row = df[df["species_name"] == species_selection].iloc[0]
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("Category", sample_row["category"])
            col_s2.metric("Population", f"{sample_row['population_count']:,}")
            col_s3.metric("Habitat Loss", f"{sample_row['habitat_loss_percent']}%")
            col_s4.metric("Status", sample_row["conservation_status"])

            sim_habitat = sample_row["habitat_loss_percent"]
            sim_pop = sample_row["population_count"]
            sim_human = sample_row["human_activity_index"]
            sim_prot = sample_row["protected_area"]
            sim_temp = sample_row["avg_temperature"]
            sim_status = sample_row["conservation_status"]
        else:
            sim_col1, sim_col2 = st.columns(2)
            with sim_col1:
                sim_habitat = st.slider("Simulated Habitat Loss (%):", 0.0, 100.0, 65.0, 1.0)
                sim_pop = st.number_input("Simulated Population Count:", 10, 100000, 350, 50)
                sim_temp = st.slider("Simulated Avg Temperature (°C):", -10.0, 45.0, 34.0, 0.5)
            with sim_col2:
                sim_human = st.slider("Simulated Human Disturbance (1-10):", 1.0, 10.0, 8.2, 0.1)
                sim_prot = st.radio("Simulated Protected Area Status:", ["No", "Yes"], horizontal=True)
                sim_status = "Critically Endangered" if sim_habitat > 60 or sim_pop < 500 else "Endangered"

        st.markdown("#### Generated Conservation Protocols:")
        rec_list = generate_conservation_recommendations(
            habitat_loss_percent=sim_habitat,
            population_count=sim_pop,
            human_activity_index=sim_human,
            protected_area=sim_prot,
            avg_temperature=sim_temp,
            predicted_status=sim_status
        )

        for rec in rec_list:
            if rec["level"] == "CRITICAL":
                st.error(f"{rec['icon']} **{rec['domain']}**: {rec['action']}")
            elif rec["level"] == "WARNING":
                st.warning(f"{rec['icon']} **{rec['domain']}**: {rec['action']}")
            else:
                st.info(f"{rec['icon']} **{rec['domain']}**: {rec['action']}")

    with tab2:
        st.subheader("🌍 Top-10 Most At-Risk Biodiversity Regions")
        st.write("Aggregated ranking based on proportion of threatened species, rate of deforestation, and anthropogenic encroachment:")

        top_regions_df = get_top_at_risk_regions(df, top_n=10)
        
        # Display styled table
        st.dataframe(
            top_regions_df,
            use_container_width=True,
            column_config={
                "Composite Risk Score": st.column_config.ProgressColumn(
                    "Threat Index (0-100)",
                    min_value=0,
                    max_value=100,
                    format="%.1f"
                ),
                "At-Risk Species Rate (%)": st.column_config.NumberColumn(
                    "At-Risk Rate (%)",
                    format="%.1f%%"
                )
            }
        )

        st.markdown("---")

        # Visual Comparison Bar Chart
        fig_risk = px.bar(
            top_regions_df,
            x="Region",
            y="Composite Risk Score",
            color="Composite Risk Score",
            color_continuous_scale="Reds",
            title="Regional Vulnerability Ranking (Composite Threat Index)",
            text="Composite Risk Score"
        )
        fig_risk.update_layout(xaxis_title="Biodiversity Hotspot", yaxis_title="Threat Score")
        st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("""
        **Strategic Policy Interventions for Priority Hotspots:**
        1. **Western Ghats & Eastern Himalayas**: Mandate ecological buffer zones around tea/coffee estates and monitor monsoon landslide corridors.
        2. **Amazon & Borneo Rainforests**: Enforce satellite radar monitoring to detect illicit logging and prioritize indigenous land tenure rights.
        3. **Sundarbans**: Reinforce mangrove seawalls, curb saline intrusion, and construct community tiger-response fences.
        """)
