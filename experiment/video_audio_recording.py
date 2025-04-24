# import cv2
# import threading
# import time
# import datetime
# import os
# import sys
# import select

# stop_recording = threading.Event()  # Event to signal audio threads to stop

# # Ensure "data" folder exists
# DATA_FOLDER = "data"
# os.makedirs(DATA_FOLDER, exist_ok=True)

# def get_timestamp():
#     """Returns the current timestamp as YYYYMMDD_HHMMSS"""
#     return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# def record_video(camera_index, group_number, experiment_number, fps=30):
#     """Records video from a specific camera at the specified FPS."""
    
#     timestamp = get_timestamp()
#     filename = os.path.join(DATA_FOLDER, f"group_{group_number}_experiment_{experiment_number}_camera_{camera_index}_{timestamp}.mp4")

#     cap = cv2.VideoCapture(camera_index)
#     if not cap.isOpened():
#         print(f"Error: Could not open camera {camera_index}.")
#         return

#     # Try setting the FPS (this may or may not work depending on the camera and driver)
#     cap.set(cv2.CAP_PROP_FPS, fps)

#     # Get the actual FPS (this may be different from the desired FPS)
#     actual_fps = cap.get(cv2.CAP_PROP_FPS)
#     print(f"Camera {camera_index} actual FPS: {actual_fps}")

#     # Get the actual resolution
#     actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     print(f"Camera {camera_index} resolution: {actual_width}x{actual_height}")

#     # Initialize video writer
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     writer = cv2.VideoWriter(filename, fourcc, actual_fps, (actual_width, actual_height))

#     print(f"Recording video from Camera {camera_index}... Saving to {filename}")

#     prev_time = time.time()
#     while not stop_recording.is_set():
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Error: Could not read frame from camera {camera_index}.")
#             break

#         current_time = time.time()
#         elapsed_time = current_time - prev_time
#         expected_time_per_frame = 1.0 / actual_fps

#         # Sleep to sync with the expected FPS
#         if elapsed_time < expected_time_per_frame:
#             time.sleep(expected_time_per_frame - elapsed_time)

#         prev_time = current_time

#         writer.write(frame)  # Write frame to file

#     cap.release()
#     writer.release()
#     print(f"Recording from Camera {camera_index} complete.")

# def capture_keyboard_input():
#     """Captures keyboard input ('q' to stop recording)."""
#     while True:
#         # Check if the user pressed 'q' to stop recording
#         if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
#             if sys.stdin.read(1) == 'q':
#                 print("Stopping recording...")
#                 stop_recording.set()  # Signal to stop the recording threads
#                 break

# def record_multiple_cameras(camera_indices, group_number, experiment_number):
#     """Records video from multiple cameras separately."""
#     threads = []
    
#     # Start a separate thread for each camera
#     for camera_index in camera_indices:
#         thread = threading.Thread(target=record_video, args=(camera_index, group_number, experiment_number, 30))
#         thread.start()
#         threads.append(thread)
    
#     # Start the keyboard input thread to capture 'q' press
#     keyboard_thread = threading.Thread(target=capture_keyboard_input)
#     keyboard_thread.start()

#     print("Recording video from multiple cameras... Press 'q' to stop.")

#     # Wait for all camera threads to finish
#     for thread in threads:
#         thread.join()

#     # Wait for the keyboard thread to finish
#     keyboard_thread.join()

#     print("All recordings complete.")

# if __name__ == "__main__":
#     camera_indices = [0, 1]  # Update with your camera indices
#     group_number = 3
#     experiment_number = "focus"

#     try:
#         record_multiple_cameras(camera_indices, group_number, experiment_number)
#     except Exception as e:
#         print(f"An error occurred: {e}")

import cv2
import threading
import time
import datetime
import os
import sys
import select
import pyaudio
import wave

stop_recording = threading.Event()  # Event to signal threads to stop

# Ensure "data" folder exists
DATA_FOLDER = "data"
os.makedirs(DATA_FOLDER, exist_ok=True)

def get_timestamp():
    """Returns the current timestamp as YYYYMMDD_HHMMSS"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def record_audio(mic_index, group_number, experiment_number, channels=1, rate=48000, chunk=1024, format=pyaudio.paInt16):
    """Records audio from a specific microphone until 'q' is pressed."""
    
    timestamp = get_timestamp()
    filename = os.path.join(DATA_FOLDER, f"group_{group_number}_experiment_{experiment_number}_mic_{mic_index}_{timestamp}.wav")

    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=format, channels=channels, rate=rate, input=True, 
                            frames_per_buffer=chunk, input_device_index=mic_index)
    except OSError:
        print(f"Error: Could not open microphone {mic_index}. Check the device index.")
        return

    print(f"Recording audio from Mic {mic_index}... Saving to {filename}")

    frames = []
    while not stop_recording.is_set():
        try:
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
        except IOError as e:
            print(f"Audio buffer overflow error (Mic {mic_index}): {e}")
            continue

    print(f"Stopping audio recording (Mic {mic_index})...")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio from Mic {mic_index} saved to {filename}")

def record_video(camera_index, group_number, experiment_number, fps=30):
    """Records video from a specific camera at the specified FPS."""
    
    timestamp = get_timestamp()
    filename = os.path.join(DATA_FOLDER, f"group_{group_number}_experiment_{experiment_number}_camera_{camera_index}_{timestamp}.mp4")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}.")
        return

    # Try setting the FPS (this may or may not work depending on the camera and driver)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Get the actual FPS (this may be different from the desired FPS)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera {camera_index} actual FPS: {actual_fps}")

    # Get the actual resolution
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {camera_index} resolution: {actual_width}x{actual_height}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, actual_fps, (actual_width, actual_height))

    print(f"Recording video from Camera {camera_index}... Saving to {filename}")

    prev_time = time.time()
    while not stop_recording.is_set():
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame from camera {camera_index}.")
            break

        current_time = time.time()
        elapsed_time = current_time - prev_time
        expected_time_per_frame = 1.0 / actual_fps

        # Sleep to sync with the expected FPS
        if elapsed_time < expected_time_per_frame:
            time.sleep(expected_time_per_frame - elapsed_time)

        prev_time = current_time

        writer.write(frame)  # Write frame to file

    cap.release()
    writer.release()
    print(f"Recording from Camera {camera_index} complete.")

def capture_keyboard_input():
    """Captures keyboard input ('q' to stop recording)."""
    while True:
        # Check if the user pressed 'q' to stop recording
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            if sys.stdin.read(1) == 'q':
                print("Stopping recording...")
                stop_recording.set()  # Signal to stop the recording threads
                break

def record_multiple_cameras_and_mics(camera_indices, mic_indices, group_number, experiment_number):
    """Records video from multiple cameras and audio from multiple microphones."""
    video_threads = []
    audio_threads = []
    
    # Start a separate thread for each camera
    for camera_index in camera_indices:
        thread = threading.Thread(target=record_video, args=(camera_index, group_number, experiment_number, 30))
        thread.start()
        video_threads.append(thread)
    
    # Start a separate thread for each microphone
    for mic_index in mic_indices:
        thread = threading.Thread(target=record_audio, args=(mic_index, group_number, experiment_number))
        thread.start()
        audio_threads.append(thread)
    
    # Start the keyboard input thread to capture 'q' press
    keyboard_thread = threading.Thread(target=capture_keyboard_input)
    keyboard_thread.start()

    print("Recording video and audio from multiple devices... Press 'q' to stop.")

    # Wait for all camera threads to finish
    for thread in video_threads:
        thread.join()

    # Wait for all audio threads to finish
    for thread in audio_threads:
        thread.join()

    # Wait for the keyboard thread to finish
    keyboard_thread.join()

    print("All recordings complete.")

if __name__ == "__main__":
    camera_indices = [0, 1]  # Update with your camera indices
    mic_indices = [0]  # Update with your microphone indices
    group_number = 11
    experiment_number = "1"

    try:
        record_multiple_cameras_and_mics(camera_indices, mic_indices, group_number, experiment_number)
    except Exception as e:
        print(f"An error occurred: {e}")
