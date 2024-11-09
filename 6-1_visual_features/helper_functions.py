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


def parse_filename(filename):
    base_name = os.path.basename(filename)
    parts = base_name.rsplit("_", maxsplit=3)
    if len(parts) == 4:
        video_id, clip_num, d, desync_ext = parts
    else:
        raise ValueError(f"Unexpected filename format: {filename}")
    desync = desync_ext.split(".")[0]  # removing .mp4
    return video_id, clip_num, desync


def extract_mouth_roi(face_frame, desired_width=128, desired_height=128):
    # Using half of ROI size
    face_width, face_height = face_frame.size  # Correct order: width first, then height

    # Modified to only select bottom half of image, rather than a specified area
    left = face_width * 1 // 4  # 25% from the left
    right = face_width * 3 // 4  # 75% from the left
    upper = face_height * 3 // 8 # 40% from the top
    lower = upper + desired_height

    if lower > face_height:
        lower = face_height
        upper = lower - desired_height
        if upper < 0:
            upper = 0
            lower = desired_height

    left, upper, right, lower = map(int, [left, upper, right, lower])
    # print(left, right, upper, lower)
    mouth_roi = face_frame.crop((left, upper, right, lower))

    # mouth_roi = mouth_roi.resize((desired_width, desired_height), Image.ANTIALIAS)

    return mouth_roi


def extract_frames(video_path, frame_rate, test_debug=False, lip_debug_dir=None):
    try:
        vr = VideoReader(video_path, ctx=cpu())
        total_frames = len(vr)
        fps = vr.get_avg_fps()
        if fps == 0:
            fps = 15
        interval = max(1, int(fps / frame_rate))
        frame_indices = list(range(0, total_frames, interval))
        frames = vr.get_batch(frame_indices).asnumpy()

        # Convert to PIL Images and extract mouth ROI
        mouth_frames = []
        for i, frame in enumerate(frames):
            pil_frame = transforms.ToPILImage()(frame)
            mouth_roi = extract_mouth_roi(pil_frame)
            mouth_frames.append(mouth_roi)

            # Save cropped mouth frames if test_debug is enabled
            if test_debug and lip_debug_dir:
                # get video id from video path
                video_id = video_path.split("/")[-1].split("_")[0]
                mouth_roi.save(os.path.join(lip_debug_dir, f"{video_id}_{i}.png"))

        return mouth_frames
    except Exception as e:
        print(f"Error extracting frames with decord from {video_path}: {e}", flush=True)
        return None
