#! /usr/bin/env python3
import os
import math
from pathlib import Path
import traceback
from typing import Optional

import click
import ray
import torch
import torchvision
import torchaudio
from einops import rearrange

from moviepy.editor import VideoFileClip

from transformers import Wav2Vec2Processor, Wav2Vec2Model

import genmo.mochi_preview.dit.joint_model.context_parallel as cp
import genmo.mochi_preview.vae.cp_conv as cp_conv
from genmo.lib.progress import get_new_progress_bar, progress_bar
from genmo.lib.utils import Timer, save_video
from genmo.mochi_preview.pipelines import DecoderModelFactory, EncoderModelFactory
from genmo.mochi_preview.vae.models import add_fourier_features, decode_latents


class GPUContext:
    def __init__(
        self,
        *,
        encoder_factory: Optional[EncoderModelFactory] = None,
        decoder_factory: Optional[DecoderModelFactory] = None,
    ):
        t = Timer()
        self.device = torch.device(f"cuda")
        if encoder_factory is not None:
            with t("load_encoder"):
                self.encoder = encoder_factory.get_model()
        if decoder_factory is not None:
            with t("load_decoder"):
                self.decoder = decoder_factory.get_model()
        t.print_stats()


def process_audio(ctx: GPUContext, vid_path: Path):
    """
    Process audio from video using Wav2Vec 2.0, matching the video duration.
    """
    with torch.inference_mode():
        # Extract audio and get video duration
        audio_path, video_duration = extract_audio_from_video(vid_path)
        
        # Process audio using wav2vec with actual video duration
        audio_embeddings = process_audio_wav2vec(
            audio_path, 
            duration=video_duration,
            use_cuda=torch.cuda.is_available()
        )

        print(audio_embeddings.shape)
        
        # Save the full embeddings (no need for duration adjustment since we match video length)
        torch.save(
            dict(audio_embeddings=audio_embeddings),
            vid_path.with_suffix(".audio_wav2vec.pt")
        )

def process_audio_wav2vec(audio_path, duration, use_cuda=True):
    """
    Process audio using Wav2Vec 2.0 model with variable duration.
    
    Args:
        audio_path (str): Path to the audio file
        duration (float): Duration to process in seconds (based on video length)
        use_cuda (bool): Whether to use CUDA for processing
    
    Returns:
        torch.Tensor: Audio embeddings from wav2vec
    """
    # Load the wav2vec model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
    
    # Load and resample audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Take first channel if stereo
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]
    
    # Calculate desired length based on actual duration
    desired_length = math.ceil(16000 * duration)  # 16kHz sampling rate
    
    # Trim or pad to match video duration
    print(f"Waveform shape: {waveform.shape[1]}\nDesired length: {desired_length}")
    if waveform.shape[1] > desired_length:
        waveform = waveform[:, :desired_length]
    else:
        pad_length = desired_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))

    waveform = waveform.squeeze()
    
    # Prepare input for wav2vec
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    if use_cuda and torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        
    return last_hidden_states

def extract_audio_from_video(video_path, output_path=None):
    """
    Extract audio from a video file and save it as MP3.
    
    Args:
        video_path (str): Path to the video file
        output_path (str, optional): Path where to save the MP3 file. 
                                   If None, saves in the same directory as video
    
    Returns:
        str: Path to the saved MP3 file
    """
    
    # If no output path specified, create one based on video path
    if output_path is None:

        output_path = os.path.splitext(video_path)[0] + '.mp3'

    video_path = str(video_path)
    output_path = str(output_path)
    
    try:
        # Load the video file
        video = VideoFileClip(video_path)
        
        # Extract the audio
        audio = video.audio
        duration = video.duration

        
        # Write the audio file
        audio.write_audiofile(output_path)
        
        # Close the video to free up resources
        video.close()
        
        return output_path, duration
        
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None



def preprocess(ctx: GPUContext, vid_path: Path, shape: str, reconstruct: bool):
    T, H, W = [int(s) for s in shape.split("x")]
    assert (T - 1) % 6 == 0, "Expected T to be 1 mod 6"
    video, _, metadata = torchvision.io.read_video(
        str(vid_path), output_format="THWC", pts_unit="secs")
    fps = metadata["video_fps"]
    video = rearrange(video, "t h w c -> c t h w")
    og_shape = video.shape
    assert video.shape[2] == H, f"Expected {vid_path} to have height {H}, got {video.shape}"
    assert video.shape[3] == W, f"Expected {vid_path} to have width {W}, got {video.shape}"
    assert video.shape[1] >= T, f"Expected {vid_path} to have at least {T} frames, got {video.shape}"
    if video.shape[1] > T:
        video = video[:, :T]
        print(f"Trimmed video from {og_shape[1]} to first {T} frames")
    video = video.unsqueeze(0)
    video = video.float() / 127.5 - 1.0
    video = video.to(ctx.device)
    video = add_fourier_features(video)

    assert video.ndim == 5
    video = cp.local_shard(video, dim=2)  # split along time dimension

    with torch.inference_mode():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ldist = ctx.encoder(video)

        print(f"{og_shape} -> {ldist.mean.shape}")
        torch.save(
            dict(mean=ldist.mean, logvar=ldist.logvar),
            vid_path.with_suffix(".latent.pt"),
        )

        process_audio(ctx, vid_path)

        if reconstruct:
            latents = ldist.sample()
            frames = decode_latents(ctx.decoder, latents)
            frames = frames.cpu().numpy()
            save_video(frames[0], str(vid_path.with_suffix(".recon.mp4")), fps=fps)


@click.command()
@click.argument("videos_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option(
    "--model_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Path to folder containing Mochi's VAE encoder and decoder weights. Download from Hugging Face: https://huggingface.co/genmo/mochi-1-preview/blob/main/encoder.safetensors and https://huggingface.co/genmo/mochi-1-preview/blob/main/decoder.safetensors",
    default="weights/",
)
@click.option("--num_gpus", default=1, help="Number of GPUs to split the encoder over")
@click.option(
    "--recon_interval", default=10, help="Reconstruct one out of every N videos (0 to disable reconstruction)"
)
@click.option("--shape", default="163x480x848", help="Shape of the video to encode")
@click.option("--overwrite", "-ow", is_flag=True, help="Overwrite existing latents")
def batch_process(
    videos_dir: Path, model_dir: Path, num_gpus: int, recon_interval: int, shape: str, overwrite: bool
) -> None:
    """Process all videos in a directory using multiple GPUs.

    Args:
        videos_dir: Directory containing input videos
        encoder_path: Path to encoder model weights
        decoder_path: Path to decoder model weights
        num_gpus: Number of GPUs to use for parallel processing
        recon_interval: Frequency of video reconstructions (0 to disable)
    """

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Get all video paths
    video_paths = list(videos_dir.glob("**/*.mp4"))
    if not video_paths:
        print(f"No MP4 files found in {videos_dir}")
        return

    preproc = GPUContext(
        encoder_factory=EncoderModelFactory(model_path=os.path.join(model_dir, "encoder.safetensors")),
        decoder_factory=DecoderModelFactory(model_path=os.path.join(model_dir, "decoder.safetensors")),
    )
    with progress_bar(type="ray_tqdm"):
        for idx, video_path in get_new_progress_bar((list(enumerate(sorted(video_paths))))):
            if str(video_path).endswith(".recon.mp4"):
                print(f"Skipping {video_path} b/c it is a reconstruction")
                continue

            print(f"Processing {video_path}")
            try:
                if video_path.with_suffix(".latent.pt").exists() and not overwrite:
                    print(f"Skipping {video_path}")
                    continue

                preprocess(
                    ctx=preproc,
                    vid_path=video_path,
                    shape=shape,
                    reconstruct=recon_interval != 0 and idx % recon_interval == 0,
                )
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {video_path}: {str(e)}")


if __name__ == "__main__":
    batch_process()
