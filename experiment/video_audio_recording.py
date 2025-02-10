import cv2
import numpy as np
import pyaudio
import wave
import threading
import keyboard

def record_audio(filename="audio_output.wav", channels=1, rate=44100, chunk=1024, format=pyaudio.paInt16):
    """Function to continuously record audio until 'q' is pressed."""
    audio = pyaudio.PyAudio()

    stream = audio.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)
    
    print("Recording audio... Press 'q' to stop.")

    frames = []

    # Loop until 'q' is pressed
    while not keyboard.is_pressed('q'):
        data = stream.read(chunk)
        frames.append(data)

    print("Stopping audio recording...")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to {filename}")


def record_multiple_cameras(camera_indices, fps=30, resolution=(640, 480), audio_filename="audio_output.wav"):
    """Function to record video from multiple cameras and audio from the microphone simultaneously."""
    num_cameras = len(camera_indices)
    if num_cameras == 0:
        print("Error: No camera indices provided.")
        return

    captures = [cv2.VideoCapture(i) for i in camera_indices]

    for i, cap in enumerate(captures):
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_indices[i]}.")
            return

    for cap in captures:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Create a VideoWriter for each camera
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    writers = [
        cv2.VideoWriter(f"output_camera_{camera_indices[i]}.mov", fourcc, fps, resolution)
        for i in range(num_cameras)
    ]

    # Start the audio recording in a separate thread
    audio_thread = threading.Thread(target=record_audio, args=(audio_filename,))
    audio_thread.start()

    print("Recording video...")
    
    while True:
        frames = []
        for i, cap in enumerate(captures):
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame from camera {camera_indices[i]}.")
                break
            frames.append(frame)
            writers[i].write(frame)  # Save each frame to its respective file

        if not frames:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video resources
    for cap in captures:
        cap.release()
    for writer in writers:
        writer.release()
    cv2.destroyAllWindows()

    print("Waiting for audio recording to finish...")
    audio_thread.join()  # Ensure audio recording completes before exiting
    print("Recording complete.")

if __name__ == "__main__":
    camera_indices = [0, 1]  # Specify USB camera indices
    try:
        record_multiple_cameras(camera_indices)
    except Exception as e:
        print(f"An error occurred: {e}")
