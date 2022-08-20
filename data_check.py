import torch
import torchvision
from scipy.io.wavfile import write


def save_data(wav, lip_pred, lip_target, save_path, cfg):
    lip_pred = (lip_pred + 127.5) * 127.5
    lip_target = (lip_target + 127.5) * 127.5
    lip_pred = lip_pred.permute(-1, 1, 2, 0).to(device="cpu", dtype=torch.uint8)
    lip_target = lip_target.permute(-1, 1, 2, 0).to(device="cpu", dtype=torch.uint8)
    torchvision.io.write_video(
        filename=str(save_path / "pred.mp4"),
        video_array=lip_pred,
        fps=cfg.model.fps,
    )
    torchvision.io.write_video(
        filename=str(save_path / "target.mp4"),
        video_array=lip_target,
        fps=cfg.model.fps,
    )

    wav = wav.to('cpu').detach().numpy().copy()
    write(str(save_path / f"input.wav"), rate=cfg.model.sampling_rate, data=wav)