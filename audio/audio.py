import json
import math
import mimetypes
import os
from datetime import timedelta
from typing import Iterable

import pandas as pd
import torch
import torchaudio
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
from moviepy.editor import VideoFileClip
from pyannote.audio import Pipeline
from pyannote.core import Annotation
from torch import Tensor

# Prepare audio file
def prepare_audio(test_video_path, output_audio_path):
  video = VideoFileClip(test_video_path)
  video.audio.write_audiofile(output_audio_path)

def transcribe_audio(output_audio_path, model_size):

  if torch.cuda.is_available():
    device = "cuda"
    compute_type = "float16"
  else:
    device = "cpu"
    compute_type = "int8"

  model = WhisperModel(model_size, device=device, compute_type=compute_type)

  segments, _ = model.transcribe(output_audio_path, language="en")

  transcriptions = []

  for segment in segments:
    item: dict = {
        "start_time": segment.start,
        "end_time": segment.end,
        "text": segment.text,
    }
    transcriptions.append(item)

  return transcriptions


def diarize_audio(output_audio_path, token):
  pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=token
    )

  if torch.cuda.is_available():
    audio = torchaudio.load(output_audio_path)

    pipeline.to(torch.device("cuda"))
    diarization = pipeline({"waveform": audio[0], "sample_rate": audio[1]})
  else:
    diarization = pipeline(output_audio_path)

  speaker_segments = []
  for segment, _, speaker in diarization.itertracks(yield_label=True):
      speaker_segments.append({
          "start_time": segment.start,
          "end_time": segment.end,
          "speaker": speaker,
      })

  return speaker_segments

def __format_seconds_to_hhmmss(seconds: float):

    temp_seconds = math.floor(seconds)
    return str(timedelta(seconds=temp_seconds))


def merge_results(transcriptions, speaker_segments):
    i = 0
    j = 0

    merged_results = []

    while i < len(transcriptions) and j < len(speaker_segments):
        tr_start = float(transcriptions[i]["start_time"])
        tr_end = float(transcriptions[i]["end_time"])
        sp_start = float(speaker_segments[j]["start_time"])
        sp_end = float(speaker_segments[j]["end_time"])

        if tr_start < sp_end and tr_end > sp_start:
            item: dict = {
                "start_time": __format_seconds_to_hhmmss(tr_start),
                "end_time": __format_seconds_to_hhmmss(tr_end),
                "speaker": speaker_segments[j]["speaker"],
                "text": transcriptions[i]["text"],
            }

            merged_results.append(item)
            i += 1
        elif tr_end <= sp_start:
            i += 1
        else:
            j += 1

    return merged_results

def save_results(merged_results, result_path, encoding):
  df = pd.DataFrame(merged_results)
  df.to_csv(result_path, index=False, encoding=encoding)


def main(test_video_path, output_audio_path, result_path, model_size, token):
  prepare_audio(test_video_path, output_audio_path)
  transcriptions = transcribe_audio(output_audio_path, model_size)
  speaker_segments = diarize_audio(output_audio_path, token)
  merged_results = merge_results(transcriptions, speaker_segments)
  save_results(merged_results, result_path, "utf-8")


if __name__ == "__main__":
# Specify path
  test_video_path = ""
  output_audio_path = ""
  result_path = "/sampledata_20241220.csv"

  # You can choose the size of the model from "tiny", "base", "small", "medium", "large"
  model_size = "base"

"""
You need access token for segmentation and speaker diarization
https://huggingface.co/pyannote/segmentation-3.0
https://huggingface.co/pyannote/speaker-diarization-3.1
"""
  token = ""


  print("\n\n\n::Start Process\n\n\n")
  main(test_video_path, output_audio_path, result_path, model_size, token)
  print("\n\n\n::End Process\n\n\n")

