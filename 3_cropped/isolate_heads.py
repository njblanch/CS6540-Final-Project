import cv2
import matplotlib.pyplot as plt
import time
import numpy as np

SHOW_FRAMES = False


def find_closest_bounding_box(faces, start_x, start_y):
    closest_box = None
    closest_distance = float('inf')

    for (x, y, w, h) in faces:
        center_x = x + w // 2
        center_y = y + h // 2
        distance = (center_x - start_x) ** 2 + (center_y - start_y) ** 2

        if distance < closest_distance:
            closest_distance = distance
            closest_box = (x, y, w, h)

    return closest_box


def process_bounding_box(frame, bounding_box):
    x, y, w, h = bounding_box
    center_x = x + w // 2
    center_y = y + h // 2

    # Determine the ROI
    if w > 256 or h > 256:
        # Resize to the largest bounding box size (maintaining aspect ratio)
        roi = frame[y:y + h, x:x + w]
        roi = cv2.resize(roi, (256, 256))
    else:
        # Grab a 256x256 chunk around the center
        x1 = max(center_x - 128, 0)
        y1 = max(center_y - 128, 0)
        x2 = min(center_x + 128, frame.shape[1])
        y2 = min(center_y + 128, frame.shape[0])

        roi = frame[y1:y2, x1:x2]

        # Resize to 256x256 if smaller
        roi = cv2.resize(roi, (256, 256))

    return roi, center_x, center_y


def show_image(roi):
    if SHOW_FRAMES:
        # Show the processed ROI
        plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show(block=False)  # Non-blocking show
        plt.pause(0.2)  # Pause for 0.2 seconds
        plt.close()  # Close the current figure


def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def process_video_videofile(start_x, start_y, video_clip, face_cascade, frame_skip=5, distance_threshold=256):
    processed_frames = []
    face_not_found_counter = 0
    last_bounding_box = None

    for frame in video_clip.iter_frames(fps=video_clip.fps, dtype='uint8'):
        # Convert the frame from RGB to BGR (OpenCV format)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Detect faces
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            face_not_found_counter = 0
            # Find the bounding box closest to the starting position
            closest_box = find_closest_bounding_box(faces, start_x, start_y)

            if closest_box is not None:
                # Process the bounding box
                roi, new_center_x, new_center_y = process_bounding_box(frame_bgr, closest_box)  # Update start_x and start_y
                if calculate_distance((new_center_x, new_center_y), (start_x, start_y)) <= distance_threshold:
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    processed_frames.append(roi_rgb)
                    last_bounding_box = closest_box
                    start_x, start_y = new_center_x, new_center_y  # Update the last known position
                    show_image(roi)
                else:
                    return None  # Reject if distance is too far

        else:
            face_not_found_counter += 1

            # Append the last valid box if can't find a face
            # No distance calculation needed since it is the same thing
            if last_bounding_box is not None:
                roi, start_x, start_y = process_bounding_box(frame_bgr, last_bounding_box)
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                processed_frames.append(roi_rgb)
                show_image(roi)

        if face_not_found_counter >= frame_skip:
            return None

    return processed_frames

