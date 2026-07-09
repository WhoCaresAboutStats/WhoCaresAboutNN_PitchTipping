import mediapipe as mp
import cv2 as cv
import joblib as jl
import argparse
import numpy as np
import pandas as pd
import os
import pathlib as pl
import sys
from tqdm import tqdm
import streamlit as st
import sklearn as skl

import matplotlib as mpl
from matplotlib import pyplot as plt

from pitch_dictionary import PITCH_NAMES, PITCH_TYPES_CUT, PITCH_TYPES

print(f"Mediapipe Version: {mp.__version__}")
print(f"OpenCV Version: {cv.__version__}")
print(f"Scikit Learn Version: {skl.__version__}")
print(f"JobLib Version: {jl.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"StreamLit Version: {st.__version__}")
print(f"MatPlotLib Version: {mpl.__version__}")

st.set_page_config(page_title="Pitch Tipping Dashboard", layout="wide")

@st.cache_data
def load_features(path):
    for _ in tqdm(range(1), desc="Loading features"):
        return pd.read_csv(path)

@st.cache_resource
def load_model(path):
    for _ in tqdm(range(1), desc="Loading model"):
        return jl.load(path)

st.title("Pitch Tipping Analysis")

features_csv = st.text_input("Features CSV", "data/features/features.csv")
model_path = st.text_input("Model path", "data/models/tipping.pkl")

if st.button("Load"):
    df = load_features(features_csv)
    model = load_model(model_path)

    st.subheader("Feature Table")
    df_display = df.copy()
    df_display["pitch_label_name"] = df_display["pitch_label"].apply(lambda x: PITCH_NAMES[x])

    st.dataframe(df.head())

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    X = df[feature_cols].values
    y = df["pitch_label"].values

    proba = model.predict_proba(X)
    classes = model.classes_

    st.subheader("Class Probability Distribution")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, cls in enumerate(classes):
        ax.plot(proba[:, i], label=cls)
    ax.legend()
    st.pyplot(fig)