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

import random

# Dataframes
players_df = pd.read_csv("names.csv")


# Input Name
#Moved into Function 4 Loop

# Check if Player Exists in DataFrame and Create Unique ID if Not (4 Trackman Uploads)
def id_query():
  is_trackman = input("Is this a Trackman Upload or Statcast Upload? (tm/sc or options): ")
  if is_trackman.lower() == "tm":
      id_decision = input("Trackman Upload Detected. Create New ID or Use Existing ID? (create/use): ")
      if id_decision.lower() == "create":
          print("Creating Unique ID for Player...")
          print("Upload Trackman CSV for Player")
          uploaded_df = pd.read_csv("TrackData.csv")
          last_name_tm = uploaded_df['Last Name'].iloc[1]
          first_name_tm = uploaded_df['First Name'].iloc[1]
          database_df = pd.read_csv("Trackman_Custom_ID_DB.csv")
          results = database_df[(database_df['First Name'] == first_name_tm) & (database_df['Last Name'] == last_name_tm)]
          new_id = f"{last_name_tm}{first_name_tm}_{random.randint(0, 9999):04d}"
          if results.empty:
              database_df.loc[len(database_df)] = [last_name_tm, first_name_tm, new_id]
              database_df.to_csv("Trackman_Custom_ID_DB.csv", index=False)
              print("ID Added to Database.")
              print("Saving CSV with Unique ID...")
              uploaded_df['Custom ID'] = new_id
              uploaded_df.to_csv(f"TrackData_{new_id}.csv", index=False)
              print("CSV Saved as: ", f"TrackData_{new_id}.csv")
          elif not results.empty:
              print(f"Player ID Already Exists in Database: {results['Custom_ID'].values[0]}")
              print(f"Enter 'use' Next Time for This Player.")
              uploaded_df['Custom ID'] = results['Custom_ID'].values[0]
              uploaded_df.to_csv(f"TrackData_{results['Custom_ID'].values[0]}.csv", index=False)
              print("CSV Saved as: ", f"TrackData_{results['Custom_ID'].values[0]}.csv")

          # add_to_db = input("Do you want to add this ID to the Database? (yes/no): ")
          # if add_to_db.lower() == "yes":
          #     print("Adding ID to Database...")
          #     database_df = pd.read_csv("Trackman_Custom_ID_DB.csv")
          #     results = database_df[(database_df['First Name'] == first_name_tm) & (database_df['Last Name'] == last_name_tm)]
          #     if results.empty:
          #       database_df.loc[len(database_df)] = [last_name_tm, first_name_tm, new_id]
          #       database_df.to_csv("Trackman_Custom_ID_DB.csv", index=False)
          #       print("ID Added to Database.")
          #     elif not results.empty:
          #       print(f"Player ID Already Exists in Database: {results['Custom ID'].values[0]}")
          #       print(f"Enter 'use' Next Time for This Player.")
          # else:
          #     print("ID Not Added to Database.")
      elif id_decision.lower() == "use":
          print("Using Existing ID for Player...")
          pitcher_search = input("Enter Player Name: ")
          pitcher_search_f, pitcher_search_l = pitcher_search.split()
          print("Searching for Player in Database...")
          db_lookup = pd.read_csv("Trackman_Custom_ID_DB.csv")
          results = db_lookup[(db_lookup['First Name'] == pitcher_search_f) & (db_lookup['Last Name'] == pitcher_search_l)]
          if not results.empty:
              print("Player Found: ", pitcher_search)
              print("Player ID: ", results['Custom ID'].values[0])
          else:
              print("Player Not Found in Database. Please Create a New ID or Check Spelling.")
  elif is_trackman.lower() == "sc":
      print("Statcast Upload Detected. Converting Name ...")
      pitcher = input("Enter Pitcher Name (First Last): ")
      first_name, last_name = pitcher.split()
      converted_name = f'{last_name}, {first_name}'
      print("Converted Name: ", converted_name)
      matched_player = players_df[players_df['player_name'] == converted_name]
      print("Matched Player Row: ", players_df['player_name'].loc[players_df['player_name'] == converted_name])
      if not matched_player.empty:
          print("HERE")
          pitcher_id = matched_player['player_id'].values[0]
          print("Player Found: ", converted_name)
          print("Player ID: ", pitcher_id)
      else:
          print("Player Not Found in Database. Please Check Spelling or Try Again.")
  elif is_trackman.lower() == "options":
      print("TrackMan Upload: tm")
      print("Statcast Upload: sc")
      #Rerunning
      id_query()
  elif is_trackman.lower() == "secret":
      print("Hopefully this will get me into college...")
  else:
      print("Invalid Input. Please Try Again.")
      id_query()

id_query()
#Probably should remove The Dude's Name ngl
