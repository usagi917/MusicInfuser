#! /usr/bin/env python3
from pathlib import Path

import re
import random
import click
import torch
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer

from genmo.mochi_preview.pipelines import T5_MODEL, T5ModelFactory, get_conditioning_for_prompts


def aist_filename_format(filename):
    pattern = r'^g[A-Z]+_s[A-Z]+_c\d+_(d\d+_)+m[A-Z]+\d+_ch\d+\.[A-Za-z0-9]+$'
    return bool(re.match(pattern, filename))

def get_genre_name(genre_code):
    genres = {
        'BR': 'Break',
        'PO': 'Pop',
        'LO': 'Lock',
        'MH': 'Middle Hip-hop',
        'LH': 'LA style Hip-hop',
        'HO': 'House',
        'WA': 'Waack',
        'KR': 'Krump',
        'JS': 'Street Jazz',
        'JB': 'Ballet Jazz'
    }
    return genres.get(genre_code, 'Unknown Genre')

def get_situation_name(situation_code):
    situations = {
        'BM': 'basic dance',
        'FM': 'advanced dance',
        'MM': 'moving camera',
        'GR': 'group dance',
        'SH': 'showcase',
        'CY': 'cypher',
        'BT': 'battle'
    }
    return situations.get(situation_code, 'unknown')

def get_camera_view(camera_code):
    camera_views = {
        '01': 'front view',
        '02': 'front-left diagonal view',
        '03': 'left side view',
        '04': 'back-left diagonal view',
        '05': 'back view',
        '06': 'back-right diagonal view',
        '07': 'right side view',
        '08': 'front-right diagonal view',
        '09': 'close-up front view from a low angle',
        '10': 'moving front view'
    }
    return camera_views.get(camera_code, 'unknown view')

def get_dancer_info(dancer_id):
    dancers = {
        '01': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'JAZZ street', 'experience': 'approx. 13 years'},
        '02': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'JAZZ street', 'experience': '18 years'},
        '03': {'gender': 'Female', 'age': '25-30 years old', 'genre': 'JAZZ street', 'experience': '15.5 years'},
        '04': {'gender': 'Male', 'age': '25-30 years old', 'genre': 'BREAK', 'experience': '13 years'},
        '05': {'gender': 'Male', 'age': '25-30 years old', 'genre': 'BREAK', 'experience': 'BREAK 10 years HIPOP 7 years'},
        '06': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'BREAK', 'experience': '6 years'},
        '07': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'JAZZ ballet', 'experience': 'JAZZ 8 years BALLET 15 years'},
        '08': {'gender': 'Female', 'age': '30-35 years old', 'genre': 'JAZZ ballet', 'experience': 'CLASSIC BALLET approx. 25 years POP approx. 10 years HOUSE 1 years BREAK 1 years CONTEMPORARY approx. 5 years'},
        '09': {'gender': 'Female', 'age': '25-30 years old', 'genre': 'JAZZ ballet', 'experience': '23 years'},
        '10': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'POP', 'experience': '6 years'},
        '11': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'POP', 'experience': '7 years'},
        '12': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'POP', 'experience': '14 years'},
        '13': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'LOCK', 'experience': 'LOCK 5 years'},
        '14': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'LOCK', 'experience': '8 years'},
        '15': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'LOCK', 'experience': '16 years'},
        '16': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'LA style HIPHOP', 'experience': '10 years'},
        '17': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'LA style HIPHOP', 'experience': '10 years'},
        '18': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'LA style HIPHOP', 'experience': '11 years'},
        '19': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'HOUSE', 'experience': '11 years'},
        '20': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'HOUSE', 'experience': '18 years'},
        '21': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'HOUSE', 'experience': '6 years'},
        '22': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'Middle HIPHOP', 'experience': '10 years'},
        '23': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'Middle HIPHOP', 'experience': '5 years'},
        '24': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'Middle HIPHOP', 'experience': '11 years'},
        '25': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'WAACK', 'experience': '9 years'},
        '26': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'WAACK', 'experience': '10 years'},
        '27': {'gender': 'Female', 'age': '20-25 years old', 'genre': 'WAACK', 'experience': '10 years'},
        '28': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'KRUMP', 'experience': '17 years'},
        '29': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'KRUMP', 'experience': '5 years'},
        '30': {'gender': 'Male', 'age': '25-30 years old', 'genre': 'KRUMP', 'experience': '15 years'},
        '31': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'HIPHOP', 'experience': '14 years'},
        '32': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'HIPHOP', 'experience': '6 years'},
        '33': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'HIPHOP', 'experience': '10 years'},
        '34': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'LOCK', 'experience': '15 years'},
        '35': {'gender': 'Male', 'age': '20-25 years old', 'genre': 'BREAK', 'experience': '12 years'}
    }
    return dancers.get(dancer_id, {'gender': 'Unknown', 'age': 'Unknown', 'genre': 'Unknown', 'experience': 'Unknown'})

def create_dance_prompt(filename, use_templates=True):
    # First parse the filename
    basename = filename.split('/')[-1]
    name = basename.replace('.txt', '')
    parts = name.split('_')
    
    genre_code = parts[0].replace('g', '')
    situation_code = parts[1].replace('s', '')
    camera_code = parts[2].replace('c', '')
    
    # Get dancer IDs
    dancer_ids = []
    for part in parts:
        if part.startswith('d'):
            dancer_ids.append(part.replace('d', ''))
    
    # Create descriptive prompt
    genre_name = get_genre_name(genre_code)
    situation_name = get_situation_name(situation_code)
    camera_view = get_camera_view(camera_code)

    
    # Build dancer descriptions
    dancer_descriptions = []
    for d_id in dancer_ids:
        dancer = get_dancer_info(d_id)
        dancer_descriptions.append(f"a professional {dancer['gender'].lower()} dancer")
    
    # Combine into final prompt
    dancers_text = ", ".join(dancer_descriptions[:-1]) + f" and {dancer_descriptions[-1]}" if len(dancer_descriptions) > 1 else dancer_descriptions[0]
    
    if not use_templates:
        prompt = f"{dancers_text} dancing {genre_name} in a {situation_name} setting in a studio with a white backdrop"
    else:
        prompt_templates = [
            "{dancers_text} dancing {genre_name} in a {situation_name} setting in a studio with a white backdrop, captured from a {camera_view}",
            "a {camera_view} video of {dancers_text} performing {genre_name} choreography against a white background in a {situation_name} scene",
            "{dancers_text} executing {genre_name} movements in a minimalist studio space in a {situation_name} setting, shot from a {camera_view}",
            "a {genre_name} dance performance by {dancers_text} in a pristine white studio, {camera_view}, {situation_name}",
        ]
        template = random.choice(prompt_templates)
        prompt = template.format(
            dancers_text=dancers_text,
            genre_name=genre_name,
            situation_name=situation_name,
            camera_view=camera_view
        )

    return prompt

@click.command()
@click.argument("captions_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--device_id", default=0, help="GPU device ID to use")
@click.option("--overwrite", "-ow", is_flag=True, help="Overwrite existing embeddings")
def process_captions(captions_dir: Path, device_id: int, overwrite=True) -> None:
    """Process all text files in a directory using T5 encoder.

    Args:
        captions_dir: Directory containing input text files
        device_id: GPU device ID to use
    """

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Get all text file paths
    text_paths = list(captions_dir.glob("**/*.txt"))
    if not text_paths:
        print(f"No text files found in {captions_dir}")
        return

    # Initialize model and tokenizer
    model_factory = T5ModelFactory()
    device = f"cuda:{device_id}"
    model = model_factory.get_model(local_rank=0, device_id=device_id, world_size=1)
    tokenizer = T5Tokenizer.from_pretrained(T5_MODEL, legacy=False)

    with tqdm(total=len(text_paths)) as pbar:
        for text_path in text_paths:
            embed_path = text_path.with_suffix(".embed_.pt")
            embed_path_2 = text_path.with_suffix(".embed.pt")
            if embed_path.exists() and not overwrite:
                pbar.write(f"Skipping {text_path} - embeddings already exist")
                continue

            pbar.write(f"Processing {text_path}")
            try:
                with open(text_path) as f:
                    text = f.read().strip()
                text_2 = "a dance video"

                if aist_filename_format(text_path.name):
                    text = create_dance_prompt(str(text_path))
                    text_2 = "a professional dancer dancing in a studio with a white backdrop"
                    print(f"Prompt: {text}")
                    print(f"Base Prompt: {text_2}")
                elif text == "":
                    text = "a YouTube dance video"

                with torch.inference_mode():
                    conditioning = get_conditioning_for_prompts(tokenizer, model, device, [text])
                torch.save(conditioning, embed_path)

                with torch.inference_mode():
                    conditioning_2 = get_conditioning_for_prompts(tokenizer, model, device, [text_2])
                torch.save(conditioning_2, embed_path_2)

            except Exception as e:
                import traceback

                traceback.print_exc()
                pbar.write(f"Error processing {text_path}: {str(e)}")

            pbar.update(1)


if __name__ == "__main__":
    process_captions()
