# B2

# imports
import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px


# read percent df
percent_df = pd.read_csv(
    "area_business_percentage.csv",
    index_col=0
)

# title
st.title("Neighborhood Similarity Explorer")

# subtitle
st.write(
    "Cluster Vancouver neighborhoods based on business composition."
)


# slider for k value
k = st.slider(
    "Choose number of clusters",
    min_value=2,
    max_value=10,
    value=4
)

# run kmeans
kmeans = KMeans(
    n_clusters=k,
    random_state=42
)

clusters = kmeans.fit_predict(percent_df)


# pca components
pca = PCA(n_components=2)

pca_data = pca.fit_transform(percent_df)


# combine everything into one dataframe
combo_df = pd.DataFrame({
    "PC1": pca_data[:,0],
    "PC2": pca_data[:,1],
    "Cluster": clusters,
    "Area": percent_df.index
})

st.subheader("K-Means Plot")

##BEGIN[ChatGPT][https://chatgpt.com/]"This is my current plotting code, can you help me make it look better. I specifically want the labels of each point to only appear when hovered over."

# plot
fig = px.scatter(
    combo_df,
    x="PC1",
    y="PC2",
    color=combo_df["Cluster"].astype(str),  
    hover_name="Area",                      
    title="Neighborhood Clusters",
    labels={
        "PC1": "Principal Component 1",
        "PC2": "Principal Component 2",
        "color": "Cluster"
    }
)

st.plotly_chart(fig, use_container_width=True)

##END[ChatGPT]


# B3

# read business_licenses
business_df = pd.read_csv("business_licenses.csv")

# get average latitude and longitude for local areas
centroids = (
    business_df
    .groupby("localarea")[["geo_point_2d.lat", "geo_point_2d.lon"]]
    .mean()
    .reset_index()
)

# count number of businesses in each local area
business_counts = (
    business_df
    .groupby("localarea")
    .size()
    .reset_index(name="BusinessCount")
)

# merge centroids into combo_df
combo_df = combo_df.merge(
    centroids,
    left_on="Area",
    right_on="localarea"
)

# merge business counts into combo_df
combo_df = combo_df.merge(
    business_counts,
    left_on="Area",
    right_on="localarea"
)

# double check merged df columns
print(combo_df.columns)

# subtitle
st.subheader("Neighborhood Map")

# plot
map_fig = px.scatter_mapbox(
    combo_df,
    lat="geo_point_2d.lat",
    lon="geo_point_2d.lon",
    color="Cluster",
    size="BusinessCount",
    hover_name="Area",
    zoom=10,
    height=700,
    color_continuous_scale="Reds",
    mapbox_style="carto-positron"
)

st.plotly_chart(
    map_fig,
    use_container_width=True
)

# B4

# subtitle
st.subheader("Cluster Assignments")

# sort df by cluster
combo_df = combo_df.sort_values("Cluster")

# show df area name and cluster
st.dataframe(
    combo_df[["Area","Cluster"]]
)

# B4 markdown is in submitted pdf