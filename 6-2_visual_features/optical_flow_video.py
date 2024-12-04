import numpy as np
import cv2
import os
import imageio


FLOW_SIZE = (32, 32)


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
    # Extract horizontal and vertical components of flow
    horizontal_flow = flow[..., 0]
    vertical_flow = flow[..., 1]

    return horizontal_flow, vertical_flow


def add_text_above_image(image, text, font, font_scale, font_color, thickness, line_type):
    # Ensure the image has 3 channels (BGR format) for compatibility with the canvas
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # Convert to BGR

    # Create a blank canvas with extra height for the text
    height, width, _ = image.shape
    canvas = np.ones((height + 40, width, 3), dtype=np.uint8) * 255

    # Place the text above the image
    cv2.putText(canvas, text, (10, 30), font, font_scale, font_color, thickness, line_type)

    # Copy the original image below the text
    canvas[40:, :] = image

    return canvas


def visualize_optical_flow(frames, output_size):
    frames_with_flow = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0  # Increase the font size for better visibility
    font_color = (255, 0, 0)  # Green color
    thickness = 2
    line_type = cv2.LINE_AA

    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]

        # Compute optical flow
        horizontal_flow, vertical_flow = compute_optical_flow(frame1, frame2)

        # Normalize the flow components for visualization
        horizontal_flow_vis = cv2.normalize(horizontal_flow, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        vertical_flow_vis = cv2.normalize(vertical_flow, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Resize frames and flow components to the desired output size
        frame1_resized = cv2.resize(frame1, output_size, interpolation=cv2.INTER_LINEAR)
        frame2_resized = cv2.resize(frame2, output_size, interpolation=cv2.INTER_LINEAR)
        horizontal_flow_vis_resized = cv2.resize(horizontal_flow_vis, output_size, interpolation=cv2.INTER_LINEAR)
        vertical_flow_vis_resized = cv2.resize(vertical_flow_vis, output_size, interpolation=cv2.INTER_LINEAR)

        # Add text above the images
        frame1_with_text = add_text_above_image(frame1_resized, "Frame 1", font, font_scale, font_color, thickness, line_type)
        frame2_with_text = add_text_above_image(frame2_resized, "Frame 2", font, font_scale, font_color, thickness, line_type)
        horizontal_flow_with_text = add_text_above_image(horizontal_flow_vis_resized, "Horizontal Flow", font, font_scale, font_color, thickness, line_type)
        vertical_flow_with_text = add_text_above_image(vertical_flow_vis_resized, "Vertical Flow", font, font_scale, font_color, thickness, line_type)

        # Stack the frames and flow components into one image
        combined_image = np.hstack([
            frame1_with_text,
            frame2_with_text,
            horizontal_flow_with_text,
            vertical_flow_with_text
        ])

        frames_with_flow.append(combined_image)

    return frames_with_flow


def create_video(frames_with_flow, output_video_path, fps=10):
    # Get the dimensions of the frames
    height, width, _ = frames_with_flow[0].shape

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame in frames_with_flow:
        out.write(frame)

    out.release()


def load_images_from_directory(directory_path):
    frames = []

    # List all files in the directory
    for filename in sorted(os.listdir(directory_path)):  # sorted ensures frames are in order
        if filename.endswith(".jpg"):  # Check if the file is a .jpg
            image_path = os.path.join(directory_path, filename)
            # Read the image and append it to the frames list
            frame = cv2.imread(image_path)
            if frame is not None:
                frames.append(frame)

    return frames


def create_gif(frames_with_flow, output_gif_path, total_duration=10):
    # Convert the frame rate to the duration of each frame (in seconds)
    num_frames = len(frames_with_flow)
    print(num_frames)
    duration_per_frame = total_duration / num_frames
    duration_per_frame = max(1, duration_per_frame)
    print(duration_per_frame)

    # Write the frames to a GIF using imageio
    imageio.mimwrite(output_gif_path, frames_with_flow, duration=duration_per_frame)

frames = load_images_from_directory("test_frames")

# Visualize the optical flow
frames_with_flow = visualize_optical_flow(frames, (256, 256))

frames_with_flow = frames_with_flow[0:20]

# Create a GIF
create_video(frames_with_flow, 'output.mp4', fps=1)
