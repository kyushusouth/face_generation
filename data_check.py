from random import random
import torch
import torchvision
from scipy.io.wavfile import write
from pathlib import Path
import skvideo.io
import numpy as np
import librosa


def save_data(wav, lip_pred, lip_target, save_path, cfg, filename):
    lip_pred = (lip_pred + 127.5) * 127.5
    lip_target = (lip_target + 127.5) * 127.5
    lip_pred = lip_pred.permute(-1, 1, 2, 0).to(device="cpu", dtype=torch.uint8)    # (T, W, H, C)
    lip_target = lip_target.permute(-1, 1, 2, 0).to(device="cpu", dtype=torch.uint8)

    torchvision.io.write_video(
        filename=f"{str(save_path / filename)}_pred.mp4",
        video_array=lip_pred,
        fps=cfg.model.fps,
    )
    wav = wav.to('cpu').detach().numpy().copy()
    write(str(f"{str(save_path / filename)}.wav"), rate=cfg.model.sampling_rate, data=wav)


def main():
    path = Path("~/face_generation/result/default/generate/2022:08:20_01-05-38/mspec80_40/test_data").expanduser()
    mov_path = list(path.glob("**/*.mp4"))
    exp_path = mov_path[10]
    print(exp_path)
    lip, _, _ = torchvision.io.read_video(str(exp_path), pts_unit="sec")    # lip : (T, W, H, C) 

    audio_path = list(path.glob("**/*.wav"))
    exp_path = audio_path[0]
    wav, fs = librosa.load(str(exp_path), sr=16000)
    breakpoint()


if __name__ == "__main__":
    main()