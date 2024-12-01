import cv2
import matplotlib.pyplot as plt

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
    # print(flow.shape)
    flow_resized = cv2.resize(flow, FLOW_SIZE, interpolation=cv2.INTER_LINEAR)

    vertical_flow = flow_resized[..., 1]
    # print(vertical_flow.shape)

    return flow_resized

image_path1 = "frame_113.jpg"  # Replace with path to your first image
image_path2 = "frame_114.jpg"  # Replace with path to your second image

frame1 = cv2.imread(image_path1)
frame2 = cv2.imread(image_path2)

# Check if the images are loaded correctly
if frame1 is None or frame2 is None:
    print("Error loading images.")
else:
    # Compute optical flow
    vertical_flow = compute_optical_flow(frame1, frame2)

    # Display the two images and the optical flow side by side
    fig, axs = plt.subplots(1, 4, figsize=(15, 6))

    # Display the first image
    axs[0].imshow(frame1)
    axs[0].set_title('Image 1')
    axs[0].axis('off')  # Hide axes

    # Display the second image
    axs[1].imshow(frame2)
    axs[1].set_title('Image 2')
    axs[1].axis('off')  # Hide axes

    # Display the optical flow
    axs[2].imshow(vertical_flow[..., 0], cmap='jet')
    axs[2].set_title('horizontal Optical Flow')
    axs[2].axis('off')  # Hide axes

    axs[3].imshow(vertical_flow[..., 1], cmap='jet')
    axs[3].set_title('vertical Optical Flow')
    axs[3].axis('off')  # Hide axes

    # Show the plot
    plt.tight_layout()
    plt.show()
