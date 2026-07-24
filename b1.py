import numpy as np
import pandas as pd
import json
import geopandas as gpd


# DATA ACQUISITION

with open("business-licences.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize([feature["properties"] for feature in data["features"]])

df["geometry"] = [feature["geometry"] for feature in data["features"]]


# CLEANING

# the only businesses I decided to retain were those with an Issued license
# I removed businesses with a Pending, Cancelled, Inactive, or Gone out of Business status because they are not currently operating and likely would unnecessarily distort the clustering
df = df[df["status"] == "Issued"].copy()

# I removed records with missing values in the localarea field as this is what I will be using as my unit of analysis, therefore, records with this as a missing field will not be of use
# I did not impute because I do not want to introduce artificial to the dataset that could distort/skew the results
df = df[df["localarea"].notna()].copy()

# because I will be observing businesstypes, I will remove records with missing businesstypes
# I will not impute this data becuase I do not want to introduce artificial to the dataset that could alter and distort the results
df = df[df["businesstype"].notna()].copy()

# count how many businesses are in each area
area_counts = df["localarea"].value_counts()
print(area_counts)

# make the minimum business count = 50 because otherwise the area can be heavily influenced by a few businesses.
min_businesses = 50

# filter through areas to find the valid areas with a min business count of 50 or above
valid_areas = area_counts[
    area_counts >= min_businesses
].index

# remove areas that not valid from df
df = df[
    df["localarea"].isin(valid_areas)
].copy()


# MATRIX

# make a matrix with area and business type
area_business_matrix = pd.crosstab(
    df["localarea"],
    df["businesstype"]
)

# normalize each row and convert to percentage so we can get percentage of businesses belonging to that type in that specific area
area_business_percentage = (
    area_business_matrix
    .div(area_business_matrix.sum(axis=1), axis=0)
    * 100
)

# print the total business type percentages of each local area to ensure they all add up to 100
print(area_business_percentage.sum(axis=1))

# preview the data to make sure everything is working how it should
print(area_business_percentage.head())

# save cleaned df
df.to_csv("business_licenses.csv", index=False)

# save percentages
area_business_percentage.to_csv("area_business_percentage.csv")