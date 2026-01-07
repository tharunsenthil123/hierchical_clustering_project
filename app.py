import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# ---------------------------------
# Page config
# ---------------------------------
st.set_page_config(
    page_title="Wine Hierarchical Clustering",
    layout="wide"
)

st.title("🍷 Wine Hierarchical Clustering App")
st.write("Hierarchical (Agglomerative) clustering using Wine dataset")

# ---------------------------------
# Debug (optional – remove later)
# ---------------------------------
# st.write("Current Directory:", os.getcwd())
# st.write("Files:", os.listdir())

# ---------------------------------
# Load saved objects
# ---------------------------------
@st.cache_resource
def load_objects():
    model = joblib.load("hierarchical_model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_objects()

# ---------------------------------
# Sidebar – Upload CSV
# ---------------------------------
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Wine CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("wine_clustering_data.csv")

# ---------------------------------
# Dataset preview
# ---------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# ---------------------------------
# Preprocessing
# ---------------------------------
df = df.fillna(df.mean())

X = df[feature_cols]
X_scaled = scaler.transform(X)

# ---------------------------------
# Apply Hierarchical Clustering
# ---------------------------------
clusters = model.fit_predict(X_scaled)
df["Cluster"] = clusters

# ---------------------------------
# Results
# ---------------------------------
st.subheader("🧩 Clustered Data")
st.dataframe(df.head())

st.subheader("📊 Cluster Distribution")
st.bar_chart(df["Cluster"].value_counts())

# ---------------------------------
# Visualization (2D)
# ---------------------------------
st.subheader("📈 Cluster Visualization")

if len(feature_cols) >= 2:
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=df["Cluster"],
        cmap="viridis"
    )
    ax.set_xlabel(feature_cols[0])
    ax.set_ylabel(feature_cols[1])
    ax.set_title("Hierarchical Clustering Result")
    plt.colorbar(scatter)
    st.pyplot(fig)
else:
    st.warning("At least 2 features required for visualization")

# ---------------------------------
# Download result
# ---------------------------------
st.subheader("⬇️ Download Clustered Output")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download Clustered CSV",
    data=csv,
    file_name="wine_clustered_output.csv",
    mime="text/csv"
)
