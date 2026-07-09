[![Spelling](https://github.com/banesullivan/README/actions/workflows/spelling.yml/badge.svg)](https://github.com/banesullivan/README/actions/workflows/spelling.yml)
# Introduction

Welcome. It has been **80** Days since my previous issue.

The timeline for this project was long, sporadic, and plagued with new ideas. I went from MediaPipe to OpenMoCap to DeepLabCut(which may be included in the next update) to anything under the sun. I'm super proud of all the work I've done to integrate a StreamLit server into this project–that was a great upgrade.

---
# Statcast Neural Network

Notes: 
- Originally I was going to do a Trackman Neural Network (may be in a future update) but I couldn't source enough info on how they save file paths so it ended up not working out
	- As a result some scripts contain Trackman features/capabilities/queries that are not fully fleshed out yet
- Large "Big Boy" Data was not uploaded due to constant errors and its pointless b/c the ability to download is there
	- The models, however, are trained from a "Big Boy" beginning on 03-15-2024 and ending on 10-30-2024

### Scripts List

| Script                              | Role                                                                                                                                                                                                                                                                          |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DownloadTheBigBoy.py                | Download all pitches from MLB games b/t a start date and end date<br>                                                                                                                                                                                                         |
| Name_2_ID_Query.py                  | Originally was going to do NN with Statcast ID not Name, but takes a name and returns ID (ASCII insensitive)                                                                                                                                                                  |
| Download_Filtered_SC.py             | Takes specific "Big Boy" + requested pitcher and outputs all pitches on the "Big Boy" csv that were thrown by the pitcher                                                                                                                                                     |
| Statcast_NeuralNetwork.py           | Takes Filtered csv and trains a model based on that (OUTDATED)                                                                                                                                                                                                                |
| Statcast_NeuralNetwork_Spin_Test.py | Found Out from Statcast csv documentation^(1)^ that two of my pitch characteristics were the same so one was irrelevant. Thus, I pivoted to spin rate as a characteristic which made it more accurate + normalize discrepancies between trained data and inputted. (USE THIS) |
| SC_Predictions.py                   | Take user input to guess pitches based on trained models. (OUTDATED)                                                                                                                                                                                                          |
| SC_Predictions_Spin_Test.py         | Similar to SC_NN_ST, uses updated characteristics. This is not compatible with models trained by the OUTDATED SC Neural Network. (USE THIS)                                                                                                                                   |
| FUTURE RELEASES-------------------- | ---------------------------------------------------------------                                                                                                                                                                                                               |
| TM_Neural_Network.py                | Trackman Neural Network                                                                                                                                                                                                                                                       |
| TM_Pitch_Guess.py                   | Trackman pitch guesser from neural-network-trained model                                                                                                                                                                                                                      |
### Miscellaneous 

| Extras                | Role + Status                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Many DBs              | Originally Databases for TM mainly to avoid making multiple custom TM IDs for players when attempting to call their files |
| Logic & TXTs          | Contains the Logic for some of the more complex scripts (Hard Scripts = I do Logic then Syntax most of the time)          |
| NoClueHowTheseGotHere | Found out later these were package versions that somehow got saved i have it in other projects just never noticed them    |
## Future Updates:
- [ ] Trackman Neural Network
	- [ ] Will Use Grip Recognition as additional characteristic for MLB pitchers
- [ ] Parquet Integration
	- [ ] Better to use column-based then row-based
- [ ] Send Me Feature Requests

---
# WhoCaresAboutPitchTipping

## Background:
In between the time of completing the first half and developing the second half of Issue 2, I took my talents to Codecademy (not sponsored their courses helped so much), no LeBron James. This one is definitely my favorite and is far more impressive. The Scripts and iterations will be detailed in a following table, but first let's examine a brief timeline, a testament to the chaos. 

### [**TIMELINE**](https://markwhen.com/whocaresaboutstats/WCAProjectMain)
==Code For Timeline is additionally located in the repository==

## Project

### Scripts List

| Scripts                       | Roles                                                                             | Run                                                |
| ----------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------- |
| Alignment_Selector.py         | Alignment of the two videos                                                       | streamlit run Scripts/Python/alignment_selector.py |
| Pitch_Tipping.py              | Many outdated things in this but mainly takes an alignment json and outputs video | python Scripts/Python/pitch_tipping.py             |
| ProtoPhase (NEED MORE ANGLES) | ----------------------                                                            | ------------------------------                     |
| Skeleton Feature              | Two Scripts that build skeleton model from frames of video                        |                                                    |
| DLC 3D Pose                   | DeepLabCut 3D Pose (currently unsupported due to lack of secondary angles)        |                                                    |
| 3+ Video Overlay integration  | Will be soon as I dont need more camera angles for this                           |                                                    |

### Requirements
- Minimum 2 videos in Data/Videos folder
- Videos preferably come from same game

Copyright Evan Arey July 2026