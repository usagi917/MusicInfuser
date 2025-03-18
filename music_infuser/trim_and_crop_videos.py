#! /usr/bin/env python3
from pathlib import Path
import shutil
import random
import math

import numpy as np
import torch

import click
from moviepy.editor import VideoFileClip
from tqdm import tqdm

@click.command()
@click.argument("folder", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_folder", type=click.Path(dir_okay=True))
@click.option("--duration", "-d", type=float, default=5.4, help="Duration in seconds")
@click.option("--resolution", "-r", type=str, default="848x480", help="Video resolution")
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["aist", "aspect"]),
    default="aist",
    help="Truncation mode: fixed (start from 0), random (random start point), segments (divide into max segments)",
)
@click.option(
    "--crop",
    "-c",
    type=click.Choice(["random", "from_start"]),
    default="random",
    help="Cropping method",
)
@click.option(
    "--num_crops",
    "-n",
    type=int,
    default=5,
    help="Number of random crops per minute",
)
@click.option(
    "--edge_exclude",
    "-e",
    type=float,
    default=0.05,
    help="Percentage of video width to exclude from edges (0.1 = 10%)",
)
def truncate_videos(folder, output_folder, duration, resolution, mode, crop, num_crops, edge_exclude):
    """Truncate all MP4 and MOV files in FOLDER to specified duration and resolution"""
    input_path = Path(folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse target resolution
    target_width, target_height = map(int, resolution.split("x"))

    # Find all MP4 and MOV files
    if mode == 'aist':
        video_files = sorted([
            f for ext in [".mp4", ".MP4", ".mov", ".MOV"]
            for f in input_path.rglob(f"*{ext}")
            if ("_c01_" in f.stem or "_c10_" in f.stem)
        ])
    video_files = (
        list(input_path.rglob("*.mp4"))
        + list(input_path.rglob("*.MOV"))
        + list(input_path.rglob("*.mov"))
        + list(input_path.rglob("*.MP4"))
    )

    for file_path in tqdm(video_files):
        try:
            relative_path = file_path.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix(".mp4")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            click.echo(f"Processing: {file_path}")
            video = VideoFileClip(str(file_path))

            # Skip if video is too short
            if video.duration < duration:
                click.echo(f"Skipping {file_path} as it is too short")
                continue

            # Skip if target resolution is larger than input
            if target_width > video.w or target_height > video.h:
                click.echo(
                    f"Skipping {file_path} as target resolution {resolution} is larger than input {video.w}x{video.h}"
                )
                continue

            segments = []
            if crop == 'from_start':
                segments = [(0, min(duration, video.duration))]
                # truncated = video.subclip(0, duration)

            elif crop == 'random':
                if video.duration <= duration:
                    segments = [(0, video.duration)]
                else:
                    # Calculate total number of crops based on video duration
                    video_minutes = video.duration / 60.0
                    total_crops = math.ceil(video_minutes * num_crops)
                    click.echo(f"Generating {total_crops} crops for {video_minutes:.1f} minutes of video")

                    # Calculate excluded time regions
                    exclude_time = video.duration * edge_exclude
                    valid_start = exclude_time
                    valid_end = video.duration - exclude_time - duration
                    
                    if valid_end <= valid_start:
                        segments = [(0, video.duration)]
                        continue
                        
                    for _ in range(total_crops):
                        start = random.uniform(valid_start, valid_end)
                        segments.append((start, start + duration))

            for idx, (start_time, end_time) in enumerate(segments):
                video.close()
                video = VideoFileClip(str(file_path))

                # Create output filename with segment index if needed
                relative_path = file_path.relative_to(input_path)
                if len(segments) > 1:
                    output_file = output_path / relative_path.with_suffix(f".seg{idx}.mp4")
                else:
                    output_file = output_path / relative_path.with_suffix(".mp4")
                
                output_file.parent.mkdir(parents=True, exist_ok=True)

                # First truncate duration
                truncated = video.subclip(start_time, end_time)

                if mode == 'aist':
                    print('Using specialized cropping for AIST')
                    # Calculate center crop coordinates
                    x1 = (video.w - 1272) // 2
                    y1 = (video.h - 720 + 180) // 2
                    
                    # Perform center crop
                    final = truncated.crop(x1=x1, y1=y1, width=1272, height=720).resize((target_width, target_height))
                else:
                    # Calculate crop dimensions to maintain aspect ratio
                    target_ratio = target_width / target_height
                    current_ratio = truncated.w / truncated.h

                    if current_ratio > target_ratio:
                        # Video is wider than target ratio - crop width
                        new_width = int(truncated.h * target_ratio)
                        x1 = (truncated.w - new_width) // 2
                        final = truncated.crop(x1=x1, width=new_width).resize((target_width, target_height))
                    else:
                        # Video is taller than target ratio - crop height
                        new_height = int(truncated.w / target_ratio)
                        y1 = (truncated.h - new_height) // 2
                        final = truncated.crop(y1=y1, height=new_height).resize((target_width, target_height))

                # Set output parameters for consistent MP4 encoding
                output_params = {
                    "codec": "libx264",
                    "audio_codec": "mp3",  # Specify audio codec explicitly
                    "audio": True,  # Enable audio
                    "preset": "medium",  # Balance between speed and quality
                    "bitrate": "5000k",  # Adjust as needed
                }
 
                # Set FPS to 30
                final = final.set_fps(30)

                # Write the output file
                final.write_videofile(str(output_file), **output_params)

                # Clean up
                truncated.close()
                final.close()

                # Check for a corresponding .txt file
                txt_file_path = file_path.with_suffix('.txt')
                if txt_file_path.exists():
                    output_txt_file = output_path / relative_path.with_suffix('.txt')
                    output_txt_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(txt_file_path, output_txt_file)
                    click.echo(f"Copied {txt_file_path} to {output_txt_file}")
                else:
                    # Print warning in bold yellow with a warning emoji
                    click.echo(f"\033[1;33m⚠️  Warning: No caption found for {file_path}, using an empty caption. This may hurt fine-tuning quality.\033[0m")
                    output_txt_file = output_path / relative_path.with_suffix('.txt')
                    output_txt_file.parent.mkdir(parents=True, exist_ok=True)
                    output_txt_file.touch()

            video.close()

        except Exception as e:
            click.echo(f"\033[1;31m Error processing {file_path}: {str(e)}\033[0m", err=True)
            try:
                if 'truncated' in locals() and truncated is not None:
                    truncated.close()
            except:
                pass
            try:
                if 'final' in locals() and final is not None:
                    final.close()
            except:
                pass
            try:
                if 'video' in locals() and video is not None:
                    video.close()
            except:
                pass
            continue


if __name__ == "__main__":
    truncate_videos()
