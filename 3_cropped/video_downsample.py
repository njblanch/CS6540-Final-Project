import subprocess
import time
from moviepy.editor import VideoFileClip


def downsample_with_moviepy(input_file, output_file, target_fps=15):
    video = VideoFileClip(input_file)

    # Downsample the video
    video_resampled = video.set_fps(target_fps)
    video_resampled.write_videofile(output_file, codec='libx264', audio_codec='aac')


def downsample_with_moviepy_no_saving(input_file, target_fps=15, min_duration=6):
    video = VideoFileClip(input_file)

    if video.duration < min_duration:
        return None

    # Downsample the video
    video_resampled = video.set_fps(target_fps)
    return video_resampled


def downsample_with_ffmpeg(input_file, output_file, target_fps=15):
    command = [
        'ffmpeg', '-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda',
        '-i', input_file, '-r', str(target_fps),
        '-c:v', 'h264_nvenc', '-c:a', 'copy', output_file
    ]

    # Call FFmpeg command to downsample
    subprocess.run(command)


if __name__=="__main__":
    input_video = 'test.mp4'
    output_video_moviepy = 'test_moviepy.mp4'
    output_video_ffmpeg = 'test_ffmpeg.mp4'

    start_time = time.time()

    # Downsample using MoviePy
    downsample_with_moviepy(input_video, output_video_moviepy)

    elapsed_time = time.time() - start_time
    print(f"MoviePy - Time taken: {elapsed_time:.2f} seconds")


    start_time = time.time()

    # Downsample using FFmpeg
    downsample_with_ffmpeg(input_video, output_video_ffmpeg)

    elapsed_time = time.time() - start_time
    print(f"FFmpeg - Time taken: {elapsed_time:.2f} seconds")


