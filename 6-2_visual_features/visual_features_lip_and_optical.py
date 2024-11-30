#!/usr/bin/env python3

import os
import sys  # Added to handle stdout
from tqdm import tqdm  # type: ignore
# import torch  # type: ignore
# import torch.nn as nn
import numpy as np  # type: ignore
from torchvision import models, transforms  # type: ignore
import csv
from collections import defaultdict
from decord import VideoReader  # type: ignore
from decord import cpu  # type: ignore
import pandas as pd  # type: ignore
import argparse
import time
import cv2
# from PIL import Image
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset
# from glob import glob
# from torch.utils.data import random_split
import insightface
import onnxruntime as ort
import subprocess
import huggingface_hub


FLOW_SIZE = (32, 32)

OMP_NUM_THREADS = 4
ONNX_MLTHREADS = 4

def load_model():
    path = huggingface_hub.hf_hub_download("public-data/insightface", "models/buffalo_l/det_10g.onnx")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    options.inter_op_num_threads = 8
    session = ort.InferenceSession(
        path, sess_options=options, providers=["CPUExecutionProvider", "CUDAExecutionProvider"]
    )
    model = insightface.model_zoo.retinaface.RetinaFace(model_file=path, session=session)
    model.input_size = (256, 256)
    return model


def parse_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.rsplit("_", maxsplit=3)
    if len(parts) == 4:
        video_id, clip_num, d, desync_ext = parts
    else:
        raise ValueError(f"Unexpected filename format: {filename}")
    desync = desync_ext.split(".")[0]  # removing .mp4
    return video_id, clip_num, desync


# Function to extract lip region using InsightFace
def extract_lip_region_from_video(vid_path, model, test_debug=False, lip_debug_dir=None):
    cap = cv2.VideoCapture(vid_path)
    lip_regions = []
    optical_flows = []

    frame_num = 0
    prev_frame = None  # To store the previous lip frame for flow computation

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # print(frame.shape)

        # Detect faces and landmarks
        faces, landmarks = model.detect(frame)

        if len(faces) == 0:
            return None

        # Get the center of the frame (image)
        frame_height, frame_width = frame.shape[:2]
        frame_center = np.array([frame_width // 2, frame_height // 2])

        # Initialize variables for selecting the closest face
        min_distance = float('inf')
        selected_face = None
        selected_landmarks = None

        # Iterate through all detected faces
        for face, lm in zip(faces, landmarks):
            # Get the bounding box of the face
            x, y = lm[2]

            # Compute the center of the face (use the center of the bounding box)
            face_center = np.array([x, y])

            # Calculate the distance from the frame center to the face center
            distance = np.linalg.norm(frame_center - face_center)

            # If this face is closer to the center, select it
            if distance < min_distance:
                min_distance = distance
                selected_face = face
                selected_landmarks = lm

        # print(landmarks)
        if selected_face is not None and selected_landmarks is not None:
            # landmarks = face.landmark
            # No face features found, reject video
            if landmarks is None:
                return None
            landmarks = landmarks[0]
            # # Extract the lip region from the landmarks (indices 48-67)
            # mouth_landmarks = landmarks[48:68]
            #
            # # Get the bounding box of the lips
            # x_min = int(np.min(mouth_landmarks[:, 0]))
            # x_max = int(np.max(mouth_landmarks[:, 0]))
            # y_min = int(np.min(mouth_landmarks[:, 1]))
            # y_max = int(np.max(mouth_landmarks[:, 1]))
            left = landmarks[3]
            right = landmarks[4]
            x_min = left[0]
            x_max = right[0]
            y_center = (left[1]+right[1])/2
            x_dist = (x_max - x_min) / 2
            y_max = int(y_center + x_dist)
            y_min = int(y_center - x_dist)
            # Crop the lip region from the frame

            lip_region = frame[y_min:y_max, int(x_min):int(x_max)]
            lip_regions.append(lip_region)

            # Optionally save cropped lip frames for debugging
            if test_debug and lip_debug_dir is not None:
                lip_region_test = cv2.cvtColor(lip_region, cv2.COLOR_BGR2GRAY)
                lip_frame_path = os.path.join(lip_debug_dir, f"frame_{frame_num}.jpg")
                cv2.imwrite(lip_frame_path, lip_region_test)

            # Compute optical flow with previous frame
            if prev_frame is not None and lip_region is not None:
                try:
                    flow = compute_optical_flow(prev_frame, lip_region)
                    optical_flows.append(flow)
                except Exception as e:
                    return None

            # Update the previous frame for the next iteration
            prev_frame = lip_region

        frame_num += 1

    cap.release()

    # # Preprocess lip regions for feature extraction
    # lip_features = []
    #
    # # Resize lip regions to a consistent shape (e.g., 128x128) and convert to tensor
    # for lip_region in lip_regions:
    #     lip_region_resized = cv2.resize(lip_region, (128, 128))  # Resize to consistent shape
    #     lip_region_tensor = transforms.ToTensor()(lip_region_resized)  # Convert to tensor
    #     lip_features.append(lip_region_tensor)

    # return lip_features, optical_flows
    return optical_flows


# Function to compute optical flow using Farneback method
def compute_optical_flow(frame1, frame2):
    # Convert the frames to grayscale
    prev_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.resize(prev_gray, FLOW_SIZE, interpolation=cv2.INTER_LINEAR)
    next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.resize(next_gray, FLOW_SIZE, interpolation=cv2.INTER_LINEAR)

    # print(next_gray.shape)

    # Compute optical flow (Farneback method)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    # print(flow.shape)
    flow_resized = cv2.resize(flow, FLOW_SIZE, interpolation=cv2.INTER_LINEAR)

    vertical_flow = flow_resized[..., 1]
    # print(vertical_flow.shape)

    return vertical_flow


def write_features_to_parquet(parquet_path, all_features):
    if not all_features:
        return
    # Determine the number of feature columns
    feature_length = len(all_features[0]) - 4
    columns = ["video_id", "clip_num", "frame_number", "desync"] + [
        f"feature_{i}" for i in range(feature_length)
    ]
    df = pd.DataFrame(all_features, columns=columns)
    df.to_parquet(parquet_path, index=False)
    print(f"Saved features to {parquet_path}", flush=True)


def main():
    print(f"\nCurrent time: {time.ctime()}", flush=True)
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 4
    sess_opts.intra_op_num_threads = 4

    result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Print the output
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error running lscpu: {result.stderr}")

    # print(f"\nCUDA available: {torch.cuda.is_available()}", flush=True)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}", flush=True)

    lip_debug_dir = None

    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract visual features from videos.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        nargs="+",
        required=True,
        help="Input directories containing videos.",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        default=False,
        help="Enable test mode.",
    )
    parser.add_argument(
        "--test_debug",
        action="store_true",
        default=False,
        help="Enable debug mode to save cropped lip frames.",
    )
    args = parser.parse_args()

    # Test debug mode flag
    test_debug = args.test_debug

    # Set output and video directories
    if args.test:
        output_dir = "/gpfs2/classes/cs6540/AVSpeech/6-2_visual_features/test_optical/"
        video_dir = "/gpfs2/classes/cs6540/AVSpeech/4_desynced/test_dist/"
    else:
        output_dir = "/gpfs2/classes/cs6540/AVSpeech/6-2_visual_features/train_optical/"
        video_dir = "/gpfs2/classes/cs6540/AVSpeech/4_desynced/train_dist/"

    # Additional directory for saving lip frames during debug
    lip_debug_dir = "./lip_region_test/"
    if test_debug:
        # allowing making this
        os.makedirs(lip_debug_dir, exist_ok=True)
        # check if folder exists, error out if it doesn't
        if not os.path.exists(lip_debug_dir):
            print(
                f"Error: Lip debug directory {lip_debug_dir} does not exist. Exiting.",
                flush=True,
            )
            sys.exit(1)
        print(
            f"Test debug mode enabled. Saving lip frames to {lip_debug_dir}", flush=True
        )

    input_dirs = args.input_dir

    # Print all input directories
    print(f"\nInput directories: {input_dirs}", flush=True)


    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput dir: {output_dir}", flush=True)

    # Getting all video ids from input dirs
    input_ids = set()
    print(f"\nFinding video ids in input directories:", flush=True)
    for input_dir in input_dirs:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".mp4"):
                    f_name = os.path.basename(file)
                    # removing anything after the last "_", as well as the _
                    if "_" in f_name:
                        f_name = f_name[: f_name.rfind("_")]
                        input_ids.add(f_name)

    print(f"Found {len(input_ids)} input ids", flush=True)

    # Finding desynced clips in video_dir
    video_ids = set()
    print(f"\nLooking for desynced clips in: {video_dir}", flush=True)
    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(".mp4"):
                f_name = os.path.basename(file)
                video_ids.add(f_name)

    print(f"Found {len(video_ids)} video ids", flush=True)

    # Filtering video files to process
    video_files = []
    print("\nChecking which videos to process", flush=True)
    for vf in video_ids:
        try:
            video_id, clip_num, desync = parse_filename(vf)
            if video_id in input_ids:
                video_files.append(vf)
        except ValueError as e:
            print(e, flush=True)

    print(f"Found {len(video_files)} videos to process", flush=True)

    print(f"\nTime after setup: {time.ctime()}", flush=True)

    frame_rate = 15
    batch_size = 256

    # Group video files by video_id
    video_dict = defaultdict(list)
    for vf in video_files:
        try:
            video_id, clip_num, desync = parse_filename(vf)
            video_dict[video_id].append((vf, clip_num, desync))
        except ValueError as e:
            print(e, flush=True)

    print(f"\nProcessing {len(video_dict)} unique video_ids", flush=True)

    model = load_model()

    # Process each video_id
    for video_id in tqdm(video_dict, desc="Processing Video IDs", file=sys.stdout):
        all_features = []
        clips = video_dict[video_id]
        for vf, clip_num, desync in clips:
            vid_path = os.path.join(video_dir, vf)
            # Extract features
            optical_flows = extract_lip_region_from_video(vid_path, model, test_debug, lip_debug_dir)

            if optical_flows is None:
                print(f"Error extracting features from: {vid_path}", flush=True)
                continue

            # Adding features and metadata
            for frame_num, feature in enumerate(optical_flows):
                flattened_flow = feature.flatten()
                all_features.append(
                    [video_id, clip_num, frame_num, desync] + flattened_flow.tolist()
                )

        if all_features and not test_debug:
            # Create a unique Parquet file name per video_id
            parquet_filename = f"{video_id}.parquet"
            parquet_path = os.path.join(output_dir, parquet_filename)
            write_features_to_parquet(parquet_path, all_features)
            print(f"Finished processing video_id: {video_id}", flush=True)
        elif test_debug:
            print(
                f"Test debug mode enabled. Skipping writing to parquet for video_id: {video_id}",
                flush=True,
            )
        else:
            print(f"No features extracted for video_id: {video_id}", flush=True)

    print(f"\nAll videos processed. Time: {time.ctime()}", flush=True)


if __name__ == "__main__":
    main()
