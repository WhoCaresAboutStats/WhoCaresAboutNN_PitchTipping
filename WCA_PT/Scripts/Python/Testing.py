import mediapipe as mp
import cv2 as cv
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
print(f"OpenCV Version: {cv.__version__}")
print(f"Scikit Learn Version: {skl.__version__}")
print(f"JobLib Version: {jl.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"StreamLit Version: {st.__version__}")
print(f"MatPlotLib Version: {mpl.__version__}")

name_test = input("File Name: ")
name_test_up = name_test.split("_")[0]
name_test_down = name_test.split("_")[1]
print(f"Name Test Up: {name_test_up}")
print(f"Name Test Down: {name_test_down}")
#IGNORE THIS STUFF I GOT CONFUSED