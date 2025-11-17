import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
df = sns.load_dataset("iris")

st.title("🌸 Iris Dataset Explorer")

st.write(
    """
This simple web app lets you explore the classic **Iris dataset**, view summary statistics, 
filter by species, and perform **KMeans clustering**.
"""
)

# Sidebar filters
st.sidebar.header("Filters")
species_filter = st.sidebar.selectbox(
    "Select species",
    options=["All"] + sorted(df["species"].unique().tolist()),
)

enable_cluster = st.sidebar.checkbox("Enable KMeans Clustering")

# Filter dataset
if species_filter != "All":
    filtered_df = df[df["species"] == species_filter]
else:
    filtered_df = df

st.subheader("📄 Dataset Preview")
st.dataframe(filtered_df)

# Summary stats
st.subheader("📊 Summary Statistics")
st.write(filtered_df.describe())

# Correlation heatmap
st.subheader("🔥 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="viridis", ax=ax)
st.pyplot(fig)

# KMeans clustering
if enable_cluster:
    st.subheader("🤖 KMeans Clustering Results")
    num_df = filtered_df.drop(columns=["species"])
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(num_df)
    filtered_df["cluster"] = clusters
    st.write(filtered_df)

    st.subheader("📍 Cluster Visualization (Petal Length vs Petal Width)")
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(
        filtered_df["petal_length"],
        filtered_df["petal_width"],
        c=clusters
    )
    ax2.set_xlabel("Petal Length")
    ax2.set_ylabel("Petal Width")
    st.pyplot(fig2)
