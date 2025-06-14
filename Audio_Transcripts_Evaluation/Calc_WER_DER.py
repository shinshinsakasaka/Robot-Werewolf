# Baseline transcript:  Group 6, Experiment 4

import pandas as pd
import jiwer
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import datetime 

# Load Data from CSV
def load_transcript_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        transcript_data = df.to_dict(orient='records')
        return transcript_data
    except FileNotFoundError:
        print(f"Error: CSV file not found at {file_path}")
        return []
    except Exception as e:
        print(f"An error occurred while reading the CSV: {e}")
        return []

# WER Calculation Function 
def calculate_wer(baseline_segments, test_segments):
    baseline_full_text = " ".join([str(seg['text']) for seg in baseline_segments])
    test_full_text = " ".join([str(seg['text']) for seg in test_segments])
    error = jiwer.wer(baseline_full_text, test_full_text)
    return error

# Helper function for DER
def parse_time(time_str):
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 3: # H:M:S
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2: # M:S
        return parts[0] * 60 + parts[1]
    else:
        raise ValueError(f"Invalid time format: {time_str}")

# Helper function for DER 
def create_pyannote_annotation(segments):
    annotation = Annotation()
    for i, seg in enumerate(segments):
        start_sec = parse_time(str(seg['start_time']))
        end_sec = parse_time(str(seg['end_time']))
        annotation[Segment(start_sec, end_sec)] = str(seg['speaker'])
    return annotation

# DER Calculation Function 
def calculate_der(baseline_segments, test_segments):
    """
    Calculates the Diarization Error Rate (DER) between baseline and test,
    and returns detailed components.
    """
    baseline_annotation = create_pyannote_annotation(baseline_segments)
    test_annotation = create_pyannote_annotation(test_segments)

    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)

    # Calling the metric object directly computes the DER and updates its internal state
    der_score = metric(baseline_annotation, test_annotation, uem=None)

    return der_score

# --- Main Execution Block ---
if __name__ == "__main__":
    baseline_csv_path = '' 
    test_csv_path = '' 

    # Load data from CSV files
    print(f"Loading baseline data from: {baseline_csv_path}")
    baseline_data = load_transcript_from_csv(baseline_csv_path)
    print(f"Loading test data from: {test_csv_path}")
    test_data = load_transcript_from_csv(test_csv_path)

    if not baseline_data or not test_data:
        print("Could not load both transcript files. Exiting.")
        exit()

    # --- WER Calculation ---
    print("\n--- WER Calculation ---")
    wer_score = calculate_wer(baseline_data, test_data)
    print(f"Calculated WER (flattened): {wer_score:.4f} ({wer_score*100:.2f}%)")

    # --- DER Calculation ---
    print("\n--- DER Calculation ---")
    der_score = calculate_der(baseline_data, test_data)
    print(f"Calculated DER: {der_score:.4f} ({der_score*100:.2f}%)")
