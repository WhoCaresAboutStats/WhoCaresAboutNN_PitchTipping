import mediapipe as mp
import cv2
import sklearn as skl
import joblib as jl
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import os
import pathlib as pl
import sys
import tqdm
import streamlit as st

print(f"Mediapipe Version: {mp.__version__}")
print(f"OpenCV Version: {cv2.__version__}")
print(f"Scikit Learn Version: {skl.__version__}")
print(f"JobLib Version: {jl.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"StreamLit Version: {st.__version__}")
print(f"MatPlotLib Version: {mpl.__version__}")