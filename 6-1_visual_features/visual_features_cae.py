#!/usr/bin/env python3

import os
import sys  # Added to handle stdout
from tqdm import tqdm  # type: ignore
import torch  # type: ignore
import torch.nn as nn
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
from PIL import Image
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from glob import glob
from torch.utils.data import random_split
from cae_train import CNN_Autoencoder
from helper_functions import extract_frames, extract_mouth_roi


def parse_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.rsplit("_", maxsplit=3)
    if len(parts) == 4:
        video_id, clip_num, d, desync_ext = parts
    else:
        raise ValueError(f"Unexpected filename format: {filename}")
    desync = desync_ext.split(".")[0]  # removing .mp4
    return video_id, clip_num, desync

def extract_features(
    path,
    model,
    preprocess,
    device,
    fps,
    batch_size=32,
    test_debug=False,
    lip_debug_dir=None,
):
    frames = extract_frames(path, fps, test_debug, lip_debug_dir)
    if frames is None:
        return None
    inputs = torch.stack([preprocess(frame) for frame in frames]).to(device)

    features = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            batch_features = model(batch)

            # flattening the features
            batch_features = batch_features.view(batch_features.size(0), -1)
            features.append(batch_features.cpu().numpy())

    features = np.concatenate(features, axis=0)

    return features


def extract_features_with_autoencoder(
    path,
    feature_extractor,
    preprocess,
    autoencoder,
    device,
    fps,
    batch_size=32,
    test_debug=False,
    lip_debug_dir=None,
):
    frames = extract_frames(path, fps, test_debug, lip_debug_dir)
    if frames is None:
        return None
    inputs = torch.stack([preprocess(frame) for frame in frames]).to(device)

    features = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            batch_features = feature_extractor(batch)
            batch_features = batch_features.view(batch_features.size(0), -1)

            # Pass through autoencoder to reduce dimensions
            compressed_features, _ = autoencoder(batch_features)
            features.append(compressed_features.cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def extract_features_using_cae(
        path,
        preprocess,
        autoencoder,
        device,
        fps,
        batch_size=32,
        test_debug=False,
        lip_debug_dir=None,
):
    frames = extract_frames(path, fps, test_debug, lip_debug_dir)
    if frames is None:
        return None
    inputs = torch.stack([preprocess(frame) for frame in frames]).to(device)

    features = []
    reconstructed_frames = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i: i + batch_size]
            compressed_features, reconstructed_batch = autoencoder(batch)

            compressed_features = compressed_features.view(compressed_features.size(0), -1)
            features.append(compressed_features.cpu().numpy())

            # Saving reconstructed frames
            if lip_debug_dir:
                # Convert reconstructed frames to numpy and save as images
                reconstructed_batch = reconstructed_batch.cpu().permute(0, 2, 3, 1).numpy()  # Convert to (N, H, W, C)

                for idx, frame in enumerate(reconstructed_batch):
                    frame_index = i + idx
                    frame_path = os.path.join(lip_debug_dir, f"frame_{idx}_{frame_index:04d}.png")
                    frame_bgr = (frame * 255).astype(np.uint8)  # Convert to 8-bit
                    cv2.imwrite(frame_path, frame_bgr)

    features = np.concatenate(features, axis=0)
    return features


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

    print(f"\nCUDA available: {torch.cuda.is_available()}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

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
        "-s",
        "--autoencoder_size",
        type=int,
        default=128,
        help="Size of the compressed features.",
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
        output_dir = "/gpfs2/classes/cs6540/AVSpeech/6-1_visual_features/test_1024_cae/"
        video_dir = "/gpfs2/classes/cs6540/AVSpeech/4_desynced/test_dist/"
    else:
        output_dir = "/gpfs2/classes/cs6540/AVSpeech/6-1_visual_features/train_1024_cae/"
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

    # if autoencoder is true, append "_autoencoder_size" to output_dir

    # remove the trailing / if it exists (handled above)
    # if output_dir.endswith("/"):
    #     output_dir = output_dir[:-1]
    # output_dir += f"_cae_{args.autoencoder_size}/"

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


    autoencoder = CNN_Autoencoder().to(device)

    # Load pretrained autoencoder
    model_path = "/gpfs1/home/s/h/sheining/videosync/cae_1024_best.pth"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        autoencoder.load_state_dict(state_dict)
    else:
        print("Pretrained autoencoder not found; proceeding without training.")
        exit()

    frame_rate = 15
    batch_size = 256

    # Initialize model
    # model = models.efficientnet_v2_s(pretrained=True)
    # model.eval()
    # model.to(device)
    #
    # # Removing the classification layer to get the feature extractor
    # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    # feature_extractor.to(device)

    # dummy_input = torch.randn(1, 3, 224, 224).to(device)
    # with torch.no_grad():
    #     output = feature_extractor(dummy_input)
    # print(f"\nThe output of the model is (1280 is non-autoencoded): {output.shape}", flush=True)
    # print(f"Model output shape: {output.shape}\n", flush=True)

    # Preprocessing steps
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(  # ImageNet standards
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Group video files by video_id
    video_dict = defaultdict(list)
    for vf in video_files:
        try:
            video_id, clip_num, desync = parse_filename(vf)
            video_dict[video_id].append((vf, clip_num, desync))
        except ValueError as e:
            print(e, flush=True)

    print(f"\nProcessing {len(video_dict)} unique video_ids", flush=True)

    # Process each video_id
    for video_id in tqdm(video_dict, desc="Processing Video IDs", file=sys.stdout):
        all_features = []
        clips = video_dict[video_id]
        for vf, clip_num, desync in clips:
            vid_path = os.path.join(video_dir, vf)
            # Extract features
            feats = extract_features_using_cae(
                vid_path,
                preprocess,
                autoencoder,
                device,
                frame_rate,
                batch_size,
                test_debug,
                lip_debug_dir,
            )

            if feats is None:
                print(f"Error extracting features from: {vid_path}", flush=True)
                continue

            # Adding features and metadata
            for frame_num, feature in enumerate(feats):
                all_features.append(
                    [video_id, clip_num, frame_num, desync] + feature.tolist()
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
