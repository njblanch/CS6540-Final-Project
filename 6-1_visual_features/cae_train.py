import torch  # type: ignore
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from glob import glob
import os
from torch.utils.data import random_split
import torch.optim as optim
from torchvision.utils import save_image
from torchvision import transforms
import random
from helper_functions import extract_frames, parse_filename


class CNN_Autoencoder(nn.Module):
    def __init__(self, input_height=128, input_width=128):
        super(CNN_Autoencoder, self).__init__()

        # Define the encoder layers
        self.encoder = nn.Sequential(
            # Uses RGB, change to 1 if using grayscale
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),  # (16, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # (32, H/4, W/4)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (64, H/8, W/8)
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),  # (16, H/16, W/16)
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),  # (16, H/32, W/32)
            nn.ReLU(),
        )

        # Calculate the size of the flattened feature map after the encoder
        self.compressed_height = input_height // 16
        self.compressed_width = input_width // 16

        # Define the decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, H/16, W/16)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, H/8, W/8)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, H/4, W/4)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),  # (8, H/2, W/2)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # (3, H, W)
            nn.Sigmoid()  # Apply Sigmoid to bring output to range [0, 1]
        )

    def forward(self, x):
        compressed = self.encoder(x)
        reconstructed = self.decoder(compressed)
        return compressed, reconstructed


class VideoFrameDataset(Dataset):
    """
    Includes downsampling of videos to a random max_videos frames.
    """
    def __init__(self, video_paths, file_basename='/gpfs2/classes/cs6540/AVSpeech/4_desynced/train_dist', frame_rate=15,
                 transform=None, test_debug=False, lip_debug_dir=None, max_videos=10000, num_frames_selected=10):
        """
        Args:
            video_paths (list[str]): Directory containing lists of files
            frame_rate (int): Frame rate to sample frames from each video.
            transform (callable, optional): Optional transform to be applied on a frame.
            test_debug (bool): Whether to enable debug mode to save frames.
            lip_debug_dir (str): Directory to save mouth ROI frames in debug mode.
        """
        self.frames_tensors = []
        self.frame_rate = frame_rate
        self.transform = transform
        self.test_debug = test_debug
        self.lip_debug_dir = lip_debug_dir
        self.num_frames_selected = num_frames_selected
        self.file_basename = file_basename

        if len(video_paths) > max_videos:
            video_paths = random.sample(video_paths, max_videos)

        for video_path in video_paths:
            video_path = os.path.join(self.file_basename, video_path)
            # Extract frames from the video
            frames = extract_frames(video_path, self.frame_rate, self.test_debug, self.lip_debug_dir)
            # if frames is None:
            #     raise ValueError(f"Error extracting frames from video: {video_path}")
            if frames is not None:
                # Select number of frames from the video
                if len(frames) > self.num_frames_selected:
                    frames = random.sample(frames, self.num_frames_selected)

                # Apply transforms if available
                if self.transform:
                    frames = [self.transform(frame) for frame in frames]
                else:
                    # Ensure the frames are converted to tensor if no transform is provided
                    frames = [transforms.ToTensor()(frame) for frame in frames]

                # Stack frames into a tensor
                frames_tensor = torch.stack(frames)  # Shape: (num_frames, C, H, W)

                self.frames_tensors.append(frames_tensor)

        print(len(self.frames_tensors))

    def __len__(self):
        return len(self.frames_tensors)

    def __getitem__(self, idx):
        # video_path = self.video_paths[idx]
        # video_path = os.path.join(self.file_basename, video_path)
        # # Extract frames from the video
        # frames = extract_frames(video_path, self.frame_rate, self.test_debug, self.lip_debug_dir)
        # # if frames is None:
        # #     raise ValueError(f"Error extracting frames from video: {video_path}")
        # if frames is not None:
        #     # Select number of frames from the video
        #     if len(frames) > self.num_frames_selected:
        #         frames = random.sample(frames, self.num_frames_selected)
        #
        #     # Apply transforms if available
        #     if self.transform:
        #         frames = [self.transform(frame) for frame in frames]
        #     else:
        #         # Ensure the frames are converted to tensor if no transform is provided
        #         frames = [transforms.ToTensor()(frame) for frame in frames]
        #
        #     # Stack frames into a tensor
        #     frames_tensor = torch.stack(frames)  # Shape: (num_frames, C, H, W)
        #
        #     return frames_tensor
        # else:
        #     return None
        return self.frames_tensors[idx]

class FrameLevelDataset(Dataset):
    # For use of splitting videos into one set of frames
    def __init__(self, video_frame_dataset):
        """
        Args:
            video_frame_dataset (Dataset): The dataset where each sample contains a batch of frames.
        """
        self.video_frame_dataset = video_frame_dataset
        self.frames = []

        # Flatten frames from each video sample into individual frames
        for video_tensor in video_frame_dataset:
            self.frames.extend(video_tensor)  # Each frame becomes an individual sample

        print("Frames in dataset", len(self.frames))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]


def train_autoencoder(autoencoder, train_dataset, test_dataset, device, num_epochs=20, learning_rate=1e-3, model_name="cae"):
    # Define optimizer and loss function
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
    # Use of MSE since each pixel level thing is continuous
    criterion = torch.nn.MSELoss()

    # Create DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    min_test_loss = 100000.0
    # Training loop
    for epoch in range(num_epochs):
        autoencoder.train()
        running_loss = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            _, reconstructed = autoencoder(inputs)
            loss = criterion(reconstructed, inputs)  # Compute reconstruction loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}", flush=True)

        # Testing phase
        autoencoder.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs.to(device)

                # Forward pass
                _, reconstructed = autoencoder(inputs)
                loss = criterion(reconstructed, inputs)  # Compute reconstruction loss
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test Loss: {avg_test_loss:.4f}", flush=True)
        if avg_test_loss < min_test_loss:
            min_test_loss = avg_test_loss
            print(f"Best test loss, saving model.")
            torch.save(autoencoder.state_dict(), f"{model_name}_best.pth")

    print("Autoencoder training and testing completed, saving model.")
    torch.save(autoencoder.state_dict(), f"{model_name}_final.pth")

    # Save reconstructed to test (only from last batch)
    save_reconstructed_images(inputs, reconstructed, epoch, "test_reconstructed")


def save_reconstructed_images(original, reconstructed, epoch, save_dir):
    """
    Save reconstructed images to files.
    Args:
        reconstructed: The reconstructed images tensor.
        epoch: The current epoch.
        save_dir: The directory to save the images.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Save a batch of reconstructed images
    # Make sure the reconstructed tensor is between 0 and 1 (for saving as image)
    reconstructed = torch.clamp(reconstructed, 0, 1)

    # Save each image as a separate file
    for i, img in enumerate(original):
        img_filename = os.path.join(save_dir, f"epoch_{epoch}_img_{i}.png")
        save_image(img, img_filename)

    for i, img in enumerate(reconstructed):
        img_filename = os.path.join(save_dir, f"epoch_{epoch}_img_{i}_reconstructed.png")
        save_image(img, img_filename)

    print(f"Saved reconstructed images for epoch {epoch} to {save_dir}")


if __name__ == "__main__":
    parent_folder = '/gpfs2/classes/cs6540/AVSpeech/2_unzipped/'
    video_dir = '/gpfs2/classes/cs6540/AVSpeech/4_desynced/train_dist'

    # List all directories that start with "xa"
    input_dirs = [
        os.path.join(parent_folder, d) for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d)) and d.startswith('xa')
    ]
    model_name = "cae_1024"
    output_dir = "train_dist_l_256/"

    # Additional directory for saving lip frames during debug
    lip_debug_dir = "./lip_region_test/"
    # input_dirs = ids_folder

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


    # For initial testing purposes
    output_dir = "test_autoencoder"
    num_epochs = 20

    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(  # ImageNet standards
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Load training dataset
    print("Loading dataset...", flush=True)
    train_dataset = VideoFrameDataset(video_files, file_basename=video_dir, transform=preprocess)

    # Train test split
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    # Make datasets at frame level
    print("Converting video dataset to frame level and loading data", flush=True)
    train_frame_dataset = FrameLevelDataset(train_dataset)
    test_frame_dataset = FrameLevelDataset(test_dataset)

    print(f"\nCUDA available: {torch.cuda.is_available()}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    autoencoder = CNN_Autoencoder()
    autoencoder.to(device)

    # Train the autoencoder
    print("Training autoencoder", flush=True)
    train_autoencoder(autoencoder, train_frame_dataset, test_frame_dataset, device, num_epochs=num_epochs, model_name=model_name)

