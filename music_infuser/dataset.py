from pathlib import Path

import random
import math

import click
import torch
from torch.utils.data import DataLoader, Dataset


def load_to_cpu(x):
    return torch.load(x, map_location=torch.device("cpu"), weights_only=True)


class LatentEmbedDataset(Dataset):
    def __init__(self, file_paths, repeat=1):
        self.items = [
            (Path(p).with_suffix(".latent.pt"), Path(p).with_suffix(".embed.pt"))
            for p in file_paths
            if Path(p).with_suffix(".latent.pt").is_file() and Path(p).with_suffix(".embed.pt").is_file()
        ]
        self.items = self.items * repeat
        print(f"Loaded {len(self.items)}/{len(file_paths)} valid file pairs.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        latent_path, embed_path = self.items[idx]
        return load_to_cpu(latent_path), load_to_cpu(embed_path)



class MusicInfuserDataset(Dataset):
    def __init__(self, file_paths, file_paths_yt=None, yt_ratio=0.0, basic_prompt_ratio=1.0, repeat=1, split='train'):
        self.items = []
        self.items_basic = []
        self.basic_prompt_ratio = basic_prompt_ratio
        self.train_cameras = ['c01', 'c10']
        self.test_tracks = ['mLH4', 'mKR2', 'mBR0', 'mLO2', 'mJB5', 'mWA0', 'mJS3', 'mMH3', 'mHO5', 'mPO1']
        for p in file_paths:
            if Path(p).with_suffix(".latent.pt").is_file() and Path(p).with_suffix(".embed.pt").is_file() and Path(p).with_suffix(f".audio_wav2vec.pt").is_file():
                file_name = Path(p).stem
                parts = file_name.split("_")
                if split == 'train':
                    if parts[2] in self.train_cameras and parts[-2] not in self.test_tracks:
                        self.items_basic.append((Path(p).with_suffix(".latent.pt"), Path(p).with_suffix(".embed.pt"), Path(p).with_suffix(f".audio_wav2vec.pt")))
            if Path(p).with_suffix(".latent.pt").is_file() and Path(p).with_suffix(".embed_.pt").is_file() and Path(p).with_suffix(f".audio_wav2vec.pt").is_file():
                file_name = Path(p).stem
                parts = file_name.split("_")
                if split == 'train':
                    if parts[2] in self.train_cameras and parts[-2] not in self.test_tracks:
                        self.items.append((Path(p).with_suffix(".latent.pt"), Path(p).with_suffix(".embed_.pt"), Path(p).with_suffix(f".audio_wav2vec.pt")))

        if file_paths_yt is not None:

            file_paths_yt = random.sample(file_paths_yt, math.ceil(len(self.items) * yt_ratio))
            for p in file_paths_yt:
                if Path(p).with_suffix(".latent.pt").is_file() and Path(p).with_suffix(".embed.pt").is_file() and Path(p).with_suffix(f".audio_wav2vec.pt").is_file():
                    self.items_basic.append((Path(p).with_suffix(".latent.pt"), Path(p).with_suffix(".embed.pt"), Path(p).with_suffix(f".audio_wav2vec.pt")))
                if Path(p).with_suffix(".latent.pt").is_file() and Path(p).with_suffix(".embed_.pt").is_file() and Path(p).with_suffix(f".audio_wav2vec.pt").is_file():
                    self.items.append((Path(p).with_suffix(".latent.pt"), Path(p).with_suffix(".embed_.pt"), Path(p).with_suffix(f".audio_wav2vec.pt")))


        self.items = self.items * repeat
        self.items_basic = self.items_basic * repeat
        print(f"Loaded {len(self.items)}/{len(file_paths)+len(file_paths_yt)} valid file pairs.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        if random.random() <= self.basic_prompt_ratio:
            latent_path, embed_path, audio_embed_path = self.items_basic[idx]
        else:
            latent_path, embed_path, audio_embed_path = self.items[idx]
        return load_to_cpu(latent_path), load_to_cpu(embed_path), load_to_cpu(audio_embed_path)



@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
def process_videos(directory):
    dir_path = Path(directory)
    mp4_files = [str(f) for f in dir_path.glob("**/*.mp4") if not f.name.endswith(".recon.mp4")]
    assert mp4_files, f"No mp4 files found"

    dataset = LatentEmbedDataset(mp4_files)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for latents, embeds in dataloader:
        print([(k, v.shape) for k, v in latents.items()])


if __name__ == "__main__":
    process_videos()
