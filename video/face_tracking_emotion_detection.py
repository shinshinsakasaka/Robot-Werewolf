import cv2
import dlib
from collections import OrderedDict
import numpy as np
import urllib.request
import os
from deepface import DeepFace
from sklearn.cluster import DBSCAN
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def download():
    model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    model_path = "shape_predictor_68_face_landmarks.dat"

    if not os.path.exists(model_path):
        print("Downloading the facial landmark predictor...")
        urllib.request.urlretrieve(model_url, model_path + ".bz2")
        import bz2
        with bz2.BZ2File(model_path + ".bz2", 'rb') as f_in:
            with open(model_path, 'wb') as f_out:
                f_out.write(f_in.read())
        print("Download complete!")
    
    return model_path

def is_same_face(face_positions, new_box, existing_id):
    """Check if new detection overlaps significantly with a tracked face"""
    if existing_id not in face_positions:
        return False
    old_box = face_positions[existing_id]
    
    # Calculate overlap area
    dx = min(new_box[2], old_box[2]) - max(new_box[0], old_box[0])
    dy = min(new_box[3], old_box[3]) - max(new_box[1], old_box[1])
    overlap_area = dx * dy if (dx > 0 and dy > 0) else 0
    
    new_area = (new_box[2]-new_box[0])*(new_box[3]-new_box[1])
    return overlap_area > 0.5 * new_area  # 50% overlap threshold

def emotion_detection(detector, cap, out):
    # Tracking variables
    trackers = OrderedDict()  # Active trackers: {face_id: tracker}
    current_face_id = 0       # Increments for each new face
    unique_face_ids = set()   # Stores ALL unique faces ever detected
    face_positions = {}       # Stores last known position of each face for re-identification

    d_emotion = {}
    d_position = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_boxes = []

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Detection every 5 frames or when no trackers exist
        if len(trackers) == 0 or (current_frame % 5) == 0:
            new_detections = OrderedDict()
            faces = detector(gray, 1)
            
            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
                box = (x1, y1, x2, y2)

                face_roi = frame[y1:y2, x1:x2]
                try:
                # Analyze just the cropped face
                    results = DeepFace.analyze(
                    face_roi, 
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip'  # Since we already detected the face
                    )
                    #emotion = results[0]['dominant_emotion']
                    happy_score = results[0]['emotion']['happy']

                except Exception as e:
                    #print(f"Emotion analysis failed: {str(e)}")
                    emotion = "unknown"
                
                # Check if this matches any existing face
                matched_id = None
                for fid in face_positions:
                    if is_same_face(face_positions, box, fid):
                        matched_id = fid
                        break
                
                if matched_id is not None:
                    # Reuse existing ID
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                    new_detections[matched_id] = tracker
                    current_boxes.append(box)
                    face_positions[matched_id] = box  # Update position

                    # d_emotion[matched_id].append((emotion, current_frame))
                    d_emotion[matched_id].append((happy_score, current_frame))
                    d_position[matched_id].append(x1)
                else:
                    # New face found
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, (x1, y1, x2-x1, y2-y1))
                    new_detections[current_face_id] = tracker
                    unique_face_ids.add(current_face_id)
                    current_boxes.append(box)
                    face_positions[current_face_id] = box

                    # d_emotion[current_face_id] = [(emotion, current_frame)]
                    d_emotion[current_face_id] = [(happy_score, current_frame)]
                    d_position[current_face_id] = [x1]

                    current_face_id += 1
            
            trackers = new_detections
        
        else:
            # Update existing trackers
            for face_id in list(trackers.keys()):
                success, bbox = trackers[face_id].update(frame)
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    box = (x, y, x+w, y+h)
                    current_boxes.append(box)
                    face_positions[face_id] = box  # Update position
                else:
                    del trackers[face_id]
                    # Note: We keep the face in unique_face_ids and face_positions
        
        # Draw boxes and IDs
        for i, (x1, y1, x2, y2) in enumerate(current_boxes):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_id = list(trackers.keys())[i] if i < len(trackers) else "?"
            cv2.putText(frame, f"ID: {face_id}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display frame count and unique faces
        cv2.putText(frame, f"Frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Total Unique Faces: {len(unique_face_ids)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Face Tracking', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return d_emotion, d_position

def clustering_faces(d_position):
    d_position_aves = {key: sum(value)/len(value) for key, value in d_position.items()}
    X = np.array(list(d_position_aves.values())).reshape(-1, 1)

    # Cluster positions (eps = 100 means positions within 100 pixels are grouped)
    clustering = DBSCAN(eps=100, min_samples=1).fit(X)
    labels = clustering.labels_

    # Group original IDs by cluster
    clustered_faces = {}
    for face_id, cluster_id in zip(d_position.keys(), labels):
        if cluster_id not in clustered_faces:
            clustered_faces[cluster_id] = []
        clustered_faces[cluster_id].append(face_id)

    print("Actual unique faces detected:", len(clustered_faces))
    print("Face ID groupings:", clustered_faces)

    return clustered_faces

def merge_emotions(clusters, emotion_data):
    merged = defaultdict(list)
    
    for cluster_id, original_ids in clusters.items():
        # Combine all emotion entries from grouped faces
        for face_id in original_ids:
            if face_id in emotion_data:
                merged[cluster_id].extend(emotion_data[face_id])
        
        # Sort by frame number (or timestamp)
        merged[cluster_id].sort(key=lambda x: x[1])
    
    return dict(merged)

def generate_plot(merged_emotions):
    plot_data = {}
    for face_id, scores in merged_emotions.items():
        frames = [frame for (score, frame) in scores]
        happy_scores = [score*1e6 for (score, frame) in scores]  # Convert to micro-units for readability
        plot_data[face_id] = (frames, happy_scores)

    plt.figure(figsize=(12, 6))
    for face_id, (frames, scores) in plot_data.items():
        plt.plot(frames, scores, '-o', label=f'Face {face_id}', alpha=0.7)

    plt.title('Happiness Scores Across All Faces')
    plt.xlabel('Frame Number')
    plt.ylabel('Happiness Score (×10⁻⁶)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main(path):

    model_path = download()
    # Initialize face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)

    # Video setup
    video_path = path
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

    d_emotion, d_position = emotion_detection(detector, cap, out)
    print(d_emotion)
    clustered_faces = clustering_faces(d_position)
    merged_emotions = merge_emotions(clustered_faces, d_emotion)
    generate_plot(merged_emotions)


if __name__ == "__main__":
    path = "/Users/shinsaka/Desktop/Python/Werewolf-robot/Data/Experiment_Data/cropped_top_half.mp4"
    main(path)
