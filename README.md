# AI Based Biodiversity Conservation Analytics 🌿📊

A comprehensive, beginner-friendly machine learning web dashboard designed to evaluate species extinction risk and assist environmental policymakers in prioritizing biodiversity conservation efforts.

---

## 🎯 Project Overview & Objectives

Biodiversity loss is accelerating globally due to habitat fragmentation, anthropogenic encroachment, and climate instability. The **AI Based Biodiversity Conservation Analytics** project provides an automated, data-driven framework to:
1. **Analyze Ecological Patterns**: Monitor demographic trends, habitat loss rates, and human interference across key global biodiversity hotspots.
2. **Predict Species Risk**: Leverage a supervised machine learning model (**Random Forest Classifier**) to classify species into IUCN Red List threat categories (*Least Concern*, *Vulnerable*, *Endangered*, *Critically Endangered*).
3. **Prescribe Conservation Interventions**: Automatically generate targeted, rule-based environmental action protocols and highlight the top 10 most endangered ecological regions.

---

## 🏗️ Project Architecture & File Structure

```text
biodiversity-analytics/
 ├── data/
 │    └── species_data.csv        # Synthetic dataset (1,500 records across 10 global hotspots)
 ├── src/
 │    ├── __init__.py
 │    ├── data_generator.py       # Reproducible synthetic dataset generator
 │    ├── preprocess.py           # Data cleaning, encoding, and feature transformation
 │    ├── model.py                # Random Forest training, evaluation, saving & inference
 │    └── insights.py             # Rule-based policy recommendations & regional threat indexing
 ├── models/
 │    └── model.pkl               # Serialized trained model bundle (Classifier, Encoders, Metrics)
 ├── app.py                       # Interactive multi-page Streamlit dashboard
 ├── requirements.txt             # Python project dependencies
 └── README.md                    # Project documentation & viva guide
```

---

## 🚀 Setup and Run Instructions

### 1. Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13)
- Windows, macOS, or Linux

### 2. Clone / Open the Project Folder
Open your terminal or command prompt in the project root:
```bash
cd "Biodiversity Analytics"
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Re-generate Dataset or Retrain Model
```bash
python src/data_generator.py
python src/model.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
After running this command, open your browser at:
👉 **`http://localhost:8501`**

---

## 📱 Dashboard Pages & Features

### 1. 📊 Data Overview
- **Metrics at a Glance**: Displays total observations (1,500), unique species count, hotspot regions, and taxonomic classes.
- **Dataset Search & Filter**: Real-time filtering by category, conservation status, and keywords (e.g. "Bengal Tiger", "Western Ghats").
- **Quality Audit**: Missing value checks and data type verification confirming 100% data completeness.
- **Descriptive Statistics**: Summary table showing mean, standard deviation, and quartiles for all numerical variables.

### 2. 📈 Exploratory Data Analysis (EDA)
- **Species Count by Category**: Bar chart showing distribution across Mammals, Birds, Reptiles, Amphibians, and Plants.
- **IUCN Threat Distribution**: Interactive donut chart showing proportions of *Least Concern*, *Vulnerable*, *Endangered*, and *Critically Endangered*.
- **Temporal Population Trends**: Multi-line chart tracking median population changes between 2015 and 2024.
- **Habitat Loss vs. Population Scatter**: Logarithmic scatter plot demonstrating how habitat destruction correlates with declining populations.
- **Ecological Correlation Heatmap**: Pearson correlation matrix analyzing interplay between temperature, human activity, habitat loss, and population.

### 3. 🗺️ Geospatial Species Map
- **Interactive OpenStreetMap Visualization**: Plots species occurrences globally using exact coordinates.
- **Status Color-Coding**: Green (Least Concern), Yellow (Vulnerable), Orange (Endangered), Red (Critically Endangered).
- **Dynamic Multi-Filters**: Filter map points dynamically by taxonomic group, region, or IUCN status.
- **Dual Map Engines**: Toggle between Plotly Interactive Maps and Folium Marker Clusters.

### 4. 🤖 AI Risk Prediction
- **Model Diagnostics**: Live display of test accuracy (~89%), Confusion Matrix heatmap, and Detailed Classification Report (Precision, Recall, F1-Score).
- **Feature Importance**: Bar chart highlighting the dominant drivers of extinction risk (Habitat Loss and Population size).
- **Real-Time Prediction Form**: Input demographic and environmental parameters (population count, habitat loss percentage, temperature, human activity index, protected area status) to get an instantaneous risk prediction with class confidence percentages.

### 5. 💡 Conservation Insights & Actions
- **Automated Policy Simulator**: Generates immediate, rule-based environmental guidelines (e.g., habitat corridors, captive breeding, anti-poaching patrols, and community compensations).
- **Top-10 At-Risk Regions**: Ranked threat table and comparative bar chart evaluating biodiversity hotspots by composite vulnerability score.

---

## 📸 Screenshots

*(Place your application screenshots in an `assets/` or `screenshots/` directory for your project report)*

| Data Overview | Exploratory Data Analysis |
| :---: | :---: |
| ![Data Overview Placeholder](assets/screenshot_data_overview.png) | ![EDA Placeholder](assets/screenshot_eda.png) |

| Geospatial Map | AI Risk Prediction |
| :---: | :---: |
| ![Map Placeholder](assets/screenshot_map.png) | ![Prediction Placeholder](assets/screenshot_prediction.png) |

---

## 🧠 Machine Learning Model: Why Random Forest?

The project uses a **Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`) for multi-class classification:

### Why Random Forest?
1. **Handles Non-Linear Relationships**: Ecological dynamics are rarely linear; a species population threshold often drops sharply only after habitat fragmentation exceeds 50%. Tree-based models capture these step changes naturally.
2. **Robust Against Overfitting**: By aggregating predictions across 100 decorrelated decision trees (bagging with bootstrap sampling and feature sub-sampling), Random Forest avoids high variance.
3. **Interpretability & Feature Importance**: Random Forest calculates Gini impurity decrease for each feature, giving conservationists transparent explanations of *why* a species is classified as at-risk.
4. **No Severe Feature Scaling Requirements**: Unlike Support Vector Machines (SVM) or Neural Networks, Random Forest is invariant to feature scale differences (e.g. population in thousands vs. human activity index from 1 to 10).

---

## 🌐 Replacing Synthetic Data with Real Datasets

To replace the synthetic dataset with real scientific observations:
1. **IUCN Red List API / CSV**:
   - Download official assessments from [IUCN Red List](https://www.iucnredlist.org/).
   - Extract `scientific_name`, `redlistCategory`, `population_trend`.
2. **GBIF (Global Biodiversity Information Facility)**:
   - Download occurrence CSVs from [GBIF.org](https://www.gbif.org/) with species coordinates (`decimalLatitude`, `decimalLongitude`) and observation dates.
3. **NASA SEDAC / Global Forest Watch**:
   - Merge satellite deforestation rasters (tree cover loss) with coordinates to calculate real `habitat_loss_percent`.
4. **Integration**:
   - Format the combined CSV to match `data/species_data.csv` column headers and re-run `python src/model.py`.

---

## 🔮 Limitations & Future Scope

### Current Limitations:
- **Tabular Environmental Proxies**: Uses regional averages for temperature and human activity rather than real-time spatial raster grids.
- **Static Snapshots**: Observations do not account for generational turnover or reproductive gestation periods.

### Future Scope:
- **Satellite Remote Sensing Integration**: Ingest real-time Sentinel-2 and Landsat multispectral imagery to compute Normalized Difference Vegetation Index (NDVI) dynamically.
- **Deep Learning Vision**: Implement Convolutional Neural Networks (e.g., YOLO or ResNet) on camera-trap images for automatic species identification.
- **Real-Time IUCN API Integration**: Synchronize live conservation category revisions directly via REST APIs.
- **Bio-Acoustic Monitoring**: Process audio clips from forest microphones using spectrogram audio transformers.

---

## 👨‍💻 Author & Academic Submission
- **Project Title:** AI Based Biodiversity Conservation Analytics
- **Degree:** B.Tech Computer Science & Engineering (Minor Project)
- **Year:** 3rd Year
