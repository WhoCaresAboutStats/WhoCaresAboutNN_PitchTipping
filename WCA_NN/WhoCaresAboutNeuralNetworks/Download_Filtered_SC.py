import torch
print(torch.__version__)

import pybaseball as pb
from pybaseball import statcast
print(pb.__version__)
from pybaseball import cache

import pandas as pd
print(pd.__version__)

import numpy as np
print(np.__version__)

import os

import math

import sys

import matplotlib as mpl
print(mpl.__version__)

# DATAFRAMES
players_df = pd.read_csv("names.csv")
pitches_df = pd.read_csv("statcast_data.csv")
local_database = pd.read_csv("SC_Local_ID_DB.csv")
folder_path_sc = "C:\\Users\\nitro\\OneDrive\\Desktop\\WhoCaresAboutStats\\WhoCaresAboutCVandKNN\\WhoCaresAboutNeuralNetworks\\filteredData"
os.makedirs(folder_path_sc, exist_ok=True)

# INPUT 4 TRAINING MODEL
pitcher = input("Enter Pitcher ID: ")

#Function to Filter DataFrame
def filter_pitcher_data():
    db_scan = local_database[local_database['ID'] == int(pitcher)]
    if db_scan.empty:
        filtered_df = pitches_df[pitches_df['pitcher'] == int(pitcher)]
        if not filtered_df.empty:
          print(f"Data filtered for pitcher ID: {pitcher}")
          last, first = filtered_df['player_name'].iloc[0].split(', ')
          confirmation = input(f"Confirm Model Training and Export for {first} {last} (y/n): ")
          if confirmation == "y":
              print(f"{first} {last}")
              full_path = os.path.join(folder_path_sc, f"filtered_data_{last}{first}.csv")
              filtered_df.to_csv(full_path, index=False)
              local_database.loc[len(local_database)] = [last, first, pitcher]
              return filtered_df
          else:
             print("Training Canceled")
             return None
        else:
          print(f"No data found for pitcher ID: {pitcher}")
          return None
    else:
       print(f"Database Match Found for {db_scan['First Name'].iloc[0]} {db_scan['Last Name'].iloc[0]}")
       print(f"You May Already Host {db_scan['Last Name'].iloc[0]}'s Files Locally. Check Your Computer And Drag the CSV entitled 'filtered_data_{db_scan['Last Name'].iloc[0]}{db_scan['First Name'].iloc[0]}'")
       twice_confirm = input(f"Do you still want to download {db_scan['Last Name'].iloc[0]}'s Files Locally? (y/n) ")
       if twice_confirm == "y":
          filtered_df = pitches_df[pitches_df['pitcher'] == int(pitcher)]
          if not filtered_df.empty:
              print(f"Data filtered for pitcher ID: {pitcher}")
              last, first = filtered_df['player_name'].iloc[0].split(', ')
              print(f"{first} {last}")
              filtered_df.to_csv(f"filtered_data_{last}{first}.csv", index=False)
              return filtered_df
          else:
              print(f"No data found for pitcher ID: {pitcher}")
              return None
       else:
            print("Training Canceled")
            return None

# Call Function to Filter Data
filter_pitcher_data()
