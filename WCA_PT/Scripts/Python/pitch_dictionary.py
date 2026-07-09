# Dictionary Taken from SC Neural Network
PITCH_TYPES_CUT = {
  "CH": 0,
  "CS": 1,
  "CU": 2,
  "EP": 3,
  "FC": 4,
  "FF": 5,
  "FO": 6,
  "FS": 7,
  "FT": 8,
  "GY": 9,
  "IN": 10,
  "KC": 11,
  "KN": 12,
  "SC": 13,
  "SI": 14,
  "SL": 15,
  "ST": 16,
  "SV": 17,
  "UN": 18
}

PITCH_TYPES = {
  "ChangeUp": 0,
  "Slow Curveball": 1,
  "Curveball": 2,
  "Eephus": 3,
  "Cut Fastball": 4,
  "Four-Seam Fastball": 5,
  "Forkball": 6,
  "Split-Fingered Fastball": 7,
  "Two-Seam Fastball": 8,
  "Gyroball": 9,
  "Intentional Ball": 10,
  "Knuckle Curveball": 11,
  "Knuckleball": 12,
  "Screwball": 13,
  "Sinking Fastball": 14,
  "Slider": 15,
  "Sweeping Curveball": 16,
  "Slurve": 17,
  "Unknown": 18
}

PITCH_NAMES_LONG = {v: k for k, v in PITCH_TYPES.items()}

PITCH_NAMES = {v: k for k, v in PITCH_TYPES_CUT.items()}

def detect_pitch_label(filename: str) -> int:
  parts = filename.split("_")
  if len(parts) < 2:
    raise ValueError(f"Filename format incorrect: {filename}")

  pitch_code = parts[1].upper()

  if pitch_code not in PITCH_TYPES_CUT:
    raise ValueError(f"Unknown pitch type '{pitch_code}' in filename: {filename}")

  return PITCH_TYPES_CUT[pitch_code]