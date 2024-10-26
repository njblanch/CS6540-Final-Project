#!/usr/bin/env python3

import os
import sys  # Added to handle stdout
from tqdm import tqdm # type: ignore
import torch # type: ignore
import numpy as np # type: ignore
from torchvision import models, transforms # type: ignore
import csv
from collections import defaultdict
from decord import VideoReader # type: ignore
from decord import cpu # type: ignore
import pandas as pd # type: ignore
import argparse
import time

def parse_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.rsplit("_", maxsplit=3)
    if len(parts) == 4:
        video_id, clip_num, d, desync_ext = parts
    else:
        raise ValueError(f"Unexpected filename format: {filename}")
    desync = desync_ext.split(".")[0]  # removing .mp4
    return video_id, clip_num, desync

class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim=1280, compressed_dim=128):
        super(Autoencoder, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, compressed_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(compressed_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, input_dim),
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return compressed, reconstructed

def extract_frames(video_path, frame_rate):
    try:
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        if fps == 0:
            fps = 15
        interval = max(1, int(fps / frame_rate))
        frame_indices = list(range(0, total_frames, interval))
        frames = vr.get_batch(frame_indices).asnumpy()
        # Convert to PIL Images
        frames = [transforms.ToPILImage()(frame) for frame in frames]
        return frames
    except Exception as e:
        print(f"Error extracting frames with decord from {video_path}: {e}", flush=True)
        return None

def extract_features(path, model, preprocess, device, fps, batch_size=32):
    frames = extract_frames(path, fps)
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
    path, feature_extractor, preprocess, autoencoder, device, fps, batch_size=32
):
    frames = extract_frames(path, fps)
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

    # Argument parsing
    parser = argparse.ArgumentParser(description="Extract visual features from videos.")
    parser.add_argument(
        "-i",
        "--input_dir",
        type=str,
        nargs='+',
        required=True,
        help="Input directories containing videos.",
    )
    parser.add_argument(
        "-t",
        "--test",
        action='store_true',
        default=False,
        help="Enable test mode.",
    )
    parser.add_argument(
        "-a",
        "--use_autoencoder",
        action='store_true',
        default=False,
        help="Use autoencoder for feature compression.",
    )
    parser.add_argument(
        "-s",
        "--autoencoder_size",
        type=int,
        default=128,
        help="Size of the compressed features.",
    )
    args = parser.parse_args()

    # if -a is true, then -s must be provided
    if args.use_autoencoder and not args.autoencoder_size:
        parser.error("Autoencoder size must be provided when using autoencoder.")

    # Define output and video directories based on test flag
    if args.test:
        output_dir = "./test_dist/"
        video_dir = "../4_desynced/test_dist/"
    else:
        output_dir = "./train_dist/"
        video_dir = "../4_desynced/train_dist/"

    input_dirs = args.input_dir

    # Print all input directories
    print(f"\nInput directories: {input_dirs}", flush=True)

    # if autoencoder is true, append "_autoencoder_size" to output_dir
    if args.use_autoencoder:
        # remove the trailing / if it exists
        if output_dir.endswith("/"):
            output_dir = output_dir[:-1]
        output_dir += f"_{args.autoencoder_size}/"

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
                        f_name = f_name[:f_name.rfind("_")]
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
    use_autoencoder = args.use_autoencoder

    # Initialize model
    model = models.efficientnet_v2_s(pretrained=True)
    model.eval()
    model.to(device)

    # Removing the classification layer to get the feature extractor
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)

    # dummy_input = torch.randn(1, 3, 224, 224).to(device)
    # with torch.no_grad():
    #     output = feature_extractor(dummy_input)
    # print(f"\nThe output of the model is (1280 is non-autoencoded): {output.shape}", flush=True)
    # print(f"Model output shape: {output.shape}\n", flush=True)

    # Preprocessing steps
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(  # ImageNet standards
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Initialize autoencoder if required
    if use_autoencoder:
        autoencoder = Autoencoder(input_dim=1280, compressed_dim=args.autoencoder_size)
        autoencoder.to(device)
    else:
        autoencoder = None

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
            if autoencoder is not None:
                feats = extract_features_with_autoencoder(
                    vid_path, feature_extractor, preprocess, autoencoder, device, frame_rate, batch_size
                )
            else:
                feats = extract_features(
                    vid_path, feature_extractor, preprocess, device, frame_rate, batch_size
                )

            if feats is None:
                print(f"Error extracting features from: {vid_path}", flush=True)
                continue

            # Adding features and metadata
            for frame_num, feature in enumerate(feats):
                all_features.append(
                    [video_id, clip_num, frame_num, desync] + feature.tolist()
                )

        if all_features:
            # Create a unique Parquet file name per video_id
            parquet_filename = f"{video_id}.parquet"
            parquet_path = os.path.join(output_dir, parquet_filename)
            write_features_to_parquet(parquet_path, all_features)
            print(f"Finished processing video_id: {video_id}", flush=True)
        else:
            print(f"No features extracted for video_id: {video_id}", flush=True)

    print(f"\nAll videos processed. Time: {time.ctime()}", flush=True)

if __name__ == "__main__":
    main()
