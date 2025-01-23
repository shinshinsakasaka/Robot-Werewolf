import cv2
import numpy as np

# Function to handle multiple cameras
def record_multiple_cameras(camera_indices, output_filename="output.avi", fps=30, resolution=(640, 640)):
    """
    Records video from multiple cameras simultaneously.

    Args:
        camera_indices: A list of camera indices (e.g., [0, 1, 2]).
        output_filename: The name of the output video file.
        fps: Frames per second for the output video.
        resolution: The resolution of the output video (width, height).
    """
    num_cameras = len(camera_indices)
    if num_cameras == 0:
        print("Error: No camera indices provided.")
        return

    # Open video capture objects for each camera
    captures = [cv2.VideoCapture(i) for i in camera_indices]

    # Check if all cameras opened successfully
    for i, cap in enumerate(captures):
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_indices[i]}.")
            return

    # Set resolution for all cameras (if possible)
    for cap in captures:
      cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
      cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])


    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed (e.g., 'MJPG', 'MP4V')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (resolution[0] * num_cameras, resolution[1]))  # Adjust width based on # of cameras

    while True:
        frames = []
        for cap in captures:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from a camera.")
                break
            frames.append(frame)

        if not frames:
            break
        
        # Combine frames horizontally
        combined_frame = np.hstack(frames)

        # Write the frame to the output video
        out.write(combined_frame)


        # Display the combined frame (optional)
        cv2.imshow("Combined Video Feed", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    for cap in captures:
        cap.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage: Record video from cameras 0, 1 and 2
if __name__ == "__main__":
    camera_indices = [0, 1, 2]  # Replace with your camera indices
    try:
        record_multiple_cameras(camera_indices)
    except Exception as e:
        print(f"An error occurred: {e}")
