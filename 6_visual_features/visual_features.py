import os
from tqdm import tqdm
import torch
import numpy as np
import torch
from torchvision import models, transforms
import csv
from collections import defaultdict
from decord import VideoReader
from decord import cpu
import pandas as pd
import argparse
import time

print(f"\nCurrent time: {time.ctime()}")

print(f"\nCude available: {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def parse_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.rsplit("_")
    # if there are more than 4 parts, it means the filename has underscores in it
    # keeping only the last 4 parts
    if len(parts) > 4:
        non_id_parts = parts[-3:]
        # then, take all but the last 3 parts as the video id
        # and concatenate them together, adding back the _
        video_id = "_".join(parts[:-3])
        clip_num, d, desync = non_id_parts
    elif len(parts) == 4:
        # print(parts)
        video_id, clip_num, d, desync = parts
    desync = desync.split(".")[0]  # removing .mp4
    return video_id, clip_num, desync

# get args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", type=str) # ../2_unzipped/***/***
# parser.add_argument("-o", "--output_dir", type=str) # 
parser.add_argument("-t", "--test", type=bool, default=False)
parser.add_argument("-a", "--use_autoencoder", type=bool, default=False)
args = parser.parse_args()

if args.test:
    output_dir = "./test_dist/"
    video_dir = "../4_desynced/test_dist/"
else:
    output_dir = "./train_dist/"
    video_dir = "../4_desynced/train_dist/"
input_dir = args.input_dir

os.makedirs(output_dir, exist_ok=True)
output_dir = output_dir[:output_dir.rfind("/")]
print(f"\nInput dir: {input_dir}")
print(f"Output dir: {output_dir}")
os.makedirs(output_dir, exist_ok=True)

# Getting all video ids from input dir
input_ids = set()
print(f"\nFinding video ids in: {input_dir}")
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(".mp4"):
            f_name = os.path.basename(file)
            # removing anything after the last "_", as well as the _
            f_name = f_name[:f_name.rfind("_")]
            input_ids.add(f_name)

print(f"Found {len(input_ids)} input ids")

# if there is any video with basename that contians that string 
# inside video_dir, add it to the set
video_ids = set()
print(f"\nLooking for desynced clips in: {video_dir}")
for root, _, files in os.walk(video_dir):
    for file in files:
        if file.endswith(".mp4"):
            f_name = os.path.basename(file)
            video_ids.add(f_name)

print(f"Found {len(video_ids)} video ids")
            
video_files = []
# getting all video_ids that contain within their name
# any input_ids
print("\nChecking which videos to process")
# using parse_filename to get the base video_id, then doing union between two sets
for vf in video_ids:
    video_id, _, _ = parse_filename(vf)
    if video_id in input_ids:
        video_files.append(vf)

print(f"Found {len(video_files)} videos to process")

print(f"\nTime after setup: {time.ctime()}")

frame_rate = 15
batch_size = 256
use_autoencoder = args.use_autoencoder

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


# Adjust the feature extraction function to pass through the autoencoder
def extract_features_with_autoencoder(
    path, feature_extractor, preprocess, autoencoder, device, fps, batch_size=32
):
    frames = extract_frames(path, fps)
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


def extract_features(path, model, preprocess, device, fps, batch_size=32):
    frames = extract_frames(path, fps)
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


def extract_frames(video_path, frame_rate):
    try:
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        if fps == 0:
            fps = 15
        interval = int(fps / frame_rate)
        frame_indices = list(range(0, total_frames, interval))
        frames = vr.get_batch(frame_indices).asnumpy()
        # Convert to PIL Images
        frames = [transforms.ToPILImage()(frame) for frame in frames]
        return frames
    except Exception as e:
        print(f"Error extracting frames with decord from {video_path}: {e}")
        return None


def write_features_to_csv(
    csv_path, video_id, clip_num, features, desync, existing=False
):
    num_features = features.shape[1]
    mode = "a" if existing else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.writer(f)
        if not existing:
            header = ["video_id", "video_number", "frame_number", "desync"] + [
                f"feature_{i}" for i in range(num_features)
            ]
            writer.writerow(header)
        for frame_num, feature in enumerate(features):
            row = [video_id, clip_num, frame_num, desync] + feature.tolist()
            writer.writerow(row)


model = models.efficientnet_v2_s(pretrained=True)
model.eval()
model.to(device)

# removing the classification to get the feature extractor
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
feature_extractor.to(device)

dummy_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    output = feature_extractor(dummy_input)

print(f"\nThe output of the model is (1280 is non-autoencoded:")
print(f"Model output shape: {output.shape}\n")

# preprocessing for ENV2
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(  # these are from ImageNet
            mean=[0.485, 0.456, 0.406],  # which is what the
            std=[0.229, 0.224, 0.225],  # model was trained on
        ),
    ]
)

if use_autoencoder:
    autoencoder = Autoencoder(input_dim=1280, compressed_dim=128)
    autoencoder.to(device)
else:
    autoencoder = None

# video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

# going across all ids
grouped_videos = defaultdict(list)
for vf in video_files:
    video_id, clip_num, desync = parse_filename(vf)
    # print(video_id, clip_num, desync)
    if video_id is not None:
        grouped_videos[video_id].append((vf, clip_num, desync))

# going over each group of clips for a given video_id
for video_id, videos in tqdm(grouped_videos.items()):
    all_features = []
    for video_file, clip_num, desync in videos:
        vid_path = os.path.join(video_dir, video_file)
        # check if autoencoder exists
        if autoencoder is not None:
            feats = extract_features_with_autoencoder(
                vid_path, feature_extractor, preprocess, autoencoder, device, frame_rate, batch_size
            )
        else:
            feats = extract_features(
                vid_path, feature_extractor, preprocess, device, frame_rate, batch_size
            )

        if feats is None:
            print(f"Error extracting features from: {vid_path}")
            continue

        # Adding features and metadata
        for frame_num, feature in enumerate(feats):
            all_features.append(
                [video_id, clip_num, frame_num, desync] + feature.tolist()
            )

    # Writing all clips to CSV at a time
    if all_features:
        columns = ["video_id", "video_number", "frame_number", "desync"] + [
            f"feature_{i}" for i in range(len(all_features[0]) - 4)
        ]
        df = pd.DataFrame(all_features, columns=columns)
        parquet_path = os.path.join(output_dir, f"{video_id}.parquet")
        # print saving location
        print(parquet_path)
        df.to_parquet(parquet_path, index=False)