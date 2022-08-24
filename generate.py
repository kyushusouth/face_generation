from fileinput import filename
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import FaceGenDataset, FaceGenTransform, get_datasets
from train import make_model
from data_check import save_data

wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def make_test_loader(cfg, data_root, mean_std_path):
    data_path = get_datasets(
        data_root=data_root,
        name=cfg.model.name,
    )
    test_trans = FaceGenTransform(cfg, "test")    
    test_dataset = FaceGenDataset(data_path, mean_std_path, test_trans, cfg, test=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=0,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return test_loader, test_dataset


def generate(cfg, gen, test_loader, dataset, device, save_path, lip_first_frame):
    gen.eval()

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, data_len = lip.to(device), feature.to(device), data_len.to(device)

        with torch.no_grad():
            out = gen(lip_first_frame, feature)
        
        label = label[0].split("_")
        label = label[:-1]
        label = "_".join(label)

        _save_path = save_path / speaker[0]
        os.makedirs(_save_path, exist_ok=True)

        save_data(
            wav=wav.squeeze(0),
            lip_pred=out.squeeze(0),
            lip_target=lip.squeeze(0),
            save_path=_save_path,
            cfg=cfg,
            filename=label,
        )

        iter_cnt += 1
        if iter_cnt > 53:
            break


def get_first_frame(cfg, exp_data_path):
    exp_frame_path = random.choice(exp_data_path)
    npz_key = np.load(str(exp_frame_path))
    lip_exp = torch.from_numpy(npz_key['lip'])
    lip_first_frame = lip_exp[..., 0].unsqueeze(0)
    lip_first_frame = (lip_first_frame - 127.5) / 127.5
    lip_first_frame = lip_first_frame.to(torch.float32)
    return lip_first_frame


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    gen, _, _, _ = make_model(cfg, device)

    model_path = Path("/home/usr4/r70264c/face_generation/check_point/default/2022:08:20_01-05-38/mspec80_40.ckpt")

    if model_path.suffix == ".ckpt":
        try:
            gen.load_state_dict(torch.load(str(model_path))['gen'])
        except:
            gen.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['gen'])
    elif model_path.suffix == ".pth":
        try:
            gen.load_state_dict(torch.load(str(model_path)))
        except:
            gen.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

    train_data_root = Path(cfg.train.lip_pre_loaded_path_9696_time_only).expanduser()
    test_data_root = Path(cfg.test.lip_pre_loaded_path_9696_time_only).expanduser()
    mean_std_path = Path(cfg.train.lip_mean_std_path_9696_time_only).expanduser()
    data_root_list = [train_data_root, test_data_root]

    save_path = Path(cfg.test.save_path).expanduser()
    save_path = save_path / model_path.parents[0].name / model_path.stem
    train_save_path = save_path / "train_data"
    test_save_path = save_path / "test_data"
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)
    save_path_list = [train_save_path, test_save_path]

    exp_frame_path = Path(cfg.train.lip_pre_loaded_path_9696_time_only).expanduser()
    exp_data_path = get_datasets(
        data_root=exp_frame_path,
        name=cfg.model.name,
    )

    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader(cfg, data_root, mean_std_path)
        lip_first_frame = get_first_frame(cfg, exp_data_path)
        generate(
            cfg=cfg,
            gen=gen,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
            lip_first_frame=lip_first_frame,
        )

if __name__ == "__main__":
    main()