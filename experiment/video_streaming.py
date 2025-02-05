import cv2
import numpy as np

def record_multiple_cameras(camera_indices, fps=30, resolution=(640, 480)):
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

    for cap in captures:
        cap.release()
    for writer in writers:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_indices = [0]
    try:
        record_multiple_cameras(camera_indices)
    except Exception as e:
        print(f"An error occurred: {e}")
