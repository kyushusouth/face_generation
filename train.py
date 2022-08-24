from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchvision

from dataset import FaceGenDataset, FaceGenTransform, get_datasets, collate_time_adjust
from model.generator import Generator
from model.discriminator import FrameDiscriminator, SequenceDiscriminator, SyncDiscriminator, FrameDiscriminatorUNet
from loss import MaskedLoss

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(
    gen, frame_disc, seq_disc, sync_disc, 
    opt_gen, opt_frame_disc, opt_sync_disc, opt_seq_disc, 
    scheduler_gen, scheduler_frame_disc, scheduler_sync_disc, scheduler_seq_disc, 
    epoch, ckpt_path):
    torch.save(
        {
            'gen': gen.state_dict(),
            'frame_disc': frame_disc.state_dict(),
            "seq_disc": seq_disc.state_dict(),
            "sync_disc": sync_disc.state_dict(),
            'opt_gen': opt_gen.state_dict(),
            "opt_frame_disc": opt_frame_disc.state_dict(),
            "opt_seq_disc": opt_seq_disc.state_dict(),
            "opt_sync_disc": opt_sync_disc.state_dict(),
            'scheduler_gen': scheduler_gen.state_dict(),
            "scheduler_frame_disc": scheduler_frame_disc.state_dict(),
            "scheduler_seq_disc": scheduler_seq_disc.state_dict(),
            "scheduler_sync_disc": scheduler_sync_disc.state_dict(),
            "random": random.getstate(),
            "np_random": np.random.get_state(), 
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            'cuda_random' : torch.cuda.get_rng_state(),
            'epoch': epoch
        }, 
        ckpt_path,
    )


def save_loss(train_loss_list, val_loss_list, save_path, filename):
    loss_save_path = save_path / f"{filename}.png"
    plt.figure()
    plt.plot(np.arange(len(train_loss_list)), train_loss_list)
    plt.plot(np.arange(len(train_loss_list)), val_loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train loss", "validation loss"])
    plt.grid()
    plt.savefig(str(loss_save_path))
    plt.close("all")
    wandb.log({f"loss {filename}": wandb.plot.line_series(
        xs=np.arange(len(train_loss_list)), 
        ys=[train_loss_list, val_loss_list],
        keys=["train loss", "validation loss"],
        title=f"{filename}",
        xname="epoch",
    )})


def make_train_val_loader(cfg, data_root, mean_std_path):
    data_path = get_datasets(
        data_root=data_root,
        name=cfg.model.name,
    )
    n_samples = len(data_path)
    train_size = int(n_samples * 0.95)
    train_data_path = data_path[:train_size]
    val_data_path = data_path[train_size:]

    train_trans = FaceGenTransform(cfg, "train")
    val_trans = FaceGenTransform(cfg, "val")

    train_dataset = FaceGenDataset(
        data_path=train_data_path,
        mean_std_path = mean_std_path,
        transform=train_trans,
        cfg=cfg,
        test=False,
    )
    val_dataset = FaceGenDataset(
        data_path=val_data_path,
        mean_std_path=mean_std_path,
        transform=val_trans,
        cfg=cfg,
        test=False,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust, cfg=cfg),
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      # 0じゃないとバグることがあります
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(collate_time_adjust, cfg=cfg),
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_model(cfg, device):
    gen = Generator(
        cfg.model.in_channels, cfg.model.img_cond_channels, cfg.model.feat_channels, 
        cfg.model.feat_cond_channels, cfg.model.noise_channels, cfg.train.gen_dropout, cfg.model.tc_ksize)
    if cfg.model.which_frame_disc == "default":
        frame_disc = FrameDiscriminator(int(cfg.model.in_channels * 2), cfg.train.frame_disc_dropout)
    elif cfg.model.which_frame_disc == "unet":
        frame_disc = FrameDiscriminatorUNet(int(cfg.model.in_channels * 2), cfg.train.frame_disc_dropout)
    seq_disc = SequenceDiscriminator(cfg.model.in_channels, cfg.model.feat_channels, cfg.train.seq_disc_dropout, cfg.train.seq_length)
    sync_disc = SyncDiscriminator(cfg.model.in_channels, cfg.model.feat_channels, cfg.train.crop_length, cfg.train.sync_disc_dropout)

    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return gen.to(device), frame_disc.to(device), seq_disc.to(device), sync_disc.to(device)


def check_movie(pred, target, cfg, filename):
    pred = (pred + 127.5) * 127.5
    target = (target + 127.5) * 127.5
    pred = pred.permute(-1, 1, 2, 0).to(device="cpu", dtype=torch.uint8)
    target = target.permute(-1, 1, 2, 0).to(device="cpu", dtype=torch.uint8)
    save_path = Path("~/face_generation/data_check").expanduser()
    save_path = save_path / current_time
    os.makedirs(save_path, exist_ok=True)
    torchvision.io.write_video(
        filename=str(save_path / f"{filename}_pred.mp4"),
        video_array=pred,
        fps=cfg.model.fps
    )
    torchvision.io.write_video(
        filename=str(save_path / f"{filename}_target.mp4"),
        video_array=target,
        fps=cfg.model.fps
    )
    wandb.log({f"{filename}_pred": wandb.Video(str(save_path / f"{filename}_pred.mp4"), fps=cfg.model.fps, format="mp4")})
    wandb.log({f"{filename}_target": wandb.Video(str(save_path / f"{filename}_target.mp4"), fps=cfg.model.fps, format="mp4")})


def train_one_epoch(gen, frame_disc, seq_disc, sync_disc, train_loader, loss_f, opt_gen, opt_frame_disc, opt_seq_disc, opt_sync_disc, device, cfg):
    epoch_loss_gen = 0
    epoch_loss_frame_disc = 0
    epoch_loss_seq_disc = 0
    epoch_loss_sync_disc = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    gen.train()
    frame_disc.train()
    seq_disc.train()
    sync_disc.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, data_len = lip.to(device), feature.to(device), data_len.to(device)

        # discriminator
        with torch.no_grad():
            out = gen(lip[..., 0], feature)     # 口唇動画の1フレーム目だけを入力し、音響特徴量から動画全体を生成させる

        # frame disc
        if int(torch.min(data_len // 2)) < lip.shape[-1]:
            frame_idx = torch.randint(0, int(torch.min(data_len // 2)), (1,))    # マスクしたフレームは使わないようにする
        else:
            frame_idx = torch.randint(0, lip.shape[-1], (1,))
        
        if cfg.model.which_frame_disc == "default":
            fake_frame = frame_disc(out[..., frame_idx].squeeze(-1), lip[..., 0])
            real_frame = frame_disc(lip[..., frame_idx].squeeze(-1), lip[..., 0])
            frame_loss = loss_f.frame_bce_loss_d(fake_frame, real_frame)
        elif cfg.model.which_frame_disc == "unet":
            fake_frame_enc, fake_frame_dec = frame_disc(out[..., frame_idx].squeeze(-1), lip[..., 0])
            real_frame_enc, real_frame_dec = frame_disc(lip[..., frame_idx].squeeze(-1), lip[..., 0])
            frame_loss = loss_f.frame_bce_loss_d(fake_frame_enc, real_frame_enc) + loss_f.frame_bce_loss_d(fake_frame_dec, real_frame_dec)
        wandb.log({"frame_loss_disc_train": frame_loss.item()})
        epoch_loss_frame_disc += frame_loss.item()

        frame_loss.backward()
        # clip_grad_norm_(frame_disc.parameters(), cfg.train.max_norm)
        opt_frame_disc.step()
        opt_frame_disc.zero_grad()

        # seq disc
        if torch.min(data_len // 2) < lip.shape[-1]:
            seq_idx = torch.randint(0, torch.min(data_len // 2) - cfg.train.seq_length, (1,))
        else:
            seq_idx = torch.randint(0, lip.shape[-1] - cfg.train.seq_length, (1,))
        fake_seq = seq_disc(out[..., seq_idx:seq_idx + cfg.train.seq_length], feature[..., seq_idx * 2:(seq_idx + cfg.train.seq_length) * 2])
        real_seq = seq_disc(lip[..., seq_idx:seq_idx + cfg.train.seq_length], feature[..., seq_idx * 2:(seq_idx + cfg.train.seq_length) * 2])
        seq_loss = loss_f.seq_bce_loss_d(fake_seq, real_seq)
        wandb.log({"seq_loss_disc_train": seq_loss.item()})
        epoch_loss_seq_disc += seq_loss.item()

        seq_loss.backward()
        # clip_grad_norm_(seq_disc.parameters(), cfg.train.max_norm)
        opt_seq_disc.step()
        opt_seq_disc.zero_grad()

        # sync disc
        # 実際に対応しているフレームの組み合わせを取得
        real_frame_idx = torch.randint(0, lip.shape[-1] - cfg.train.crop_length, (1,))
        lip_crop = lip[..., real_frame_idx:real_frame_idx + cfg.train.crop_length]
        out_crop = out[..., real_frame_idx:real_frame_idx + cfg.train.crop_length]
        feature_crop_real = feature[..., real_frame_idx * 2:(real_frame_idx + cfg.train.crop_length) * 2]

        # 間違っているフレームの組み合わせを取得
        if real_frame_idx < lip.shape[-1] // 2:
            fake_frame_idx = torch.randint(lip.shape[-1] // 2, lip.shape[-1] - cfg.train.crop_length, (1,))
        else:
            fake_frame_idx = torch.randint(0, lip.shape[-1] // 2 - cfg.train.crop_length, (1,))
        feature_crop_fake = feature[..., fake_frame_idx * 2:(fake_frame_idx + cfg.train.crop_length) * 2]

        sync_real_lip = sync_disc(lip_crop, feature_crop_real)
        sync_real_out = sync_disc(out_crop, feature_crop_real)
        sync_fake = sync_disc(lip_crop, feature_crop_fake)
        sync_loss = loss_f.sync_bce_loss_d(sync_real_lip, sync_real_out, sync_fake)
        wandb.log({"sync_loss_disc_train": sync_loss.item()})
        epoch_loss_sync_disc += sync_loss.item()

        sync_loss.backward()
        # clip_grad_norm_(sync_disc.parameters(), cfg.train.max_norm)
        opt_sync_disc.step()
        opt_sync_disc.zero_grad()

        # generator
        if cfg.train.gen_opt_step:
            out = gen(lip[..., 0], feature)

            with torch.no_grad():
                # frame disc
                if int(torch.min(data_len // 2)) < lip.shape[-1]:
                    frame_idx = torch.randint(0, int(torch.min(data_len // 2)), (1,))
                else:
                    frame_idx = torch.randint(0, lip.shape[-1], (1,))
                
                if cfg.model.which_frame_disc == "default":
                    fake_frame = frame_disc(out[..., frame_idx].squeeze(-1), lip[..., 0])
                elif cfg.model.which_frame_disc == "unet":
                    fake_frame_enc, fake_frame_dec = frame_disc(out[..., frame_idx].squeeze(-1), lip[..., 0])

                # seq disc
                if torch.min(data_len // 2) < lip.shape[-1]:
                    seq_idx = torch.randint(0, torch.min(data_len // 2) - cfg.train.seq_length, (1,))
                else:
                    seq_idx = torch.randint(0, lip.shape[-1] - cfg.train.seq_length, (1,))
                fake_seq = seq_disc(out[..., seq_idx:seq_idx + cfg.train.seq_length], feature[..., seq_idx * 2:(seq_idx + cfg.train.seq_length) * 2])

                # sync disc
                real_frame_idx = torch.randint(0, lip.shape[-1] - cfg.train.crop_length, (1,))
                out_crop = out[..., real_frame_idx:real_frame_idx + cfg.train.crop_length]
                feature_crop_real = feature[..., real_frame_idx * 2:(real_frame_idx + cfg.train.crop_length) * 2]
                sync_real_out = sync_disc(out_crop, feature_crop_real)
            
            if cfg.model.which_frame_disc == "default":
                frame_loss = loss_f.frame_bce_loss_g(fake_frame)
            elif cfg.model.which_frame_disc == "unet":
                frame_loss = loss_f.frame_bce_loss_g(fake_frame_enc) + loss_f.frame_bce_loss_g(fake_frame_dec)
            seq_loss = loss_f.seq_bce_loss_g(fake_seq)
            sync_loss = loss_f.sync_bce_loss_g(sync_real_out)
            l1_loss = loss_f.l1_loss(out, lip, data_len, lip.shape[-1])
            gen_loss = cfg.train.frame_weight * frame_loss \
                + cfg.train.seq_weight * seq_loss + cfg.train.sync_weight * sync_loss + cfg.train.l1_weight * l1_loss
            epoch_loss_gen += gen_loss.item()
            wandb.log({"frame_loss_gen_train": frame_loss.item()})
            wandb.log({"seq_loss_gen_train": seq_loss.item()})
            wandb.log({"sync_loss_gen_train": sync_loss.item()})
            wandb.log({"l1_loss_gen_train": l1_loss.item()})
            wandb.log({"loss_gen_train": gen_loss.item()})
            gen_loss.backward()
            # clip_grad_norm_(gen.parameters(), cfg.train.max_norm)
            opt_gen.step()
            opt_gen.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_movie(out[0], lip[0], cfg, "lip_train")
                break

        if iter_cnt > (all_iter - 1):
            check_movie(out[0], lip[0], cfg, "lip_train")
    
    epoch_loss_gen /= iter_cnt
    epoch_loss_frame_disc /= iter_cnt
    epoch_loss_seq_disc /= iter_cnt
    epoch_loss_sync_disc /= iter_cnt
    return epoch_loss_gen, epoch_loss_frame_disc, epoch_loss_seq_disc, epoch_loss_sync_disc


def calc_val_loss(gen, frame_disc, seq_disc, sync_disc, val_loader, loss_f, device, cfg):
    epoch_loss_gen = 0
    epoch_loss_frame_disc = 0
    epoch_loss_seq_disc = 0
    epoch_loss_sync_disc = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    gen.eval()
    frame_disc.eval()
    seq_disc.eval()
    sync_disc.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, data_len = lip.to(device), feature.to(device), data_len.to(device)

        with torch.no_grad():
            out = gen(lip[..., 0], feature) 

            if int(torch.min(data_len // 2)) < lip.shape[-1]:
                frame_idx = torch.randint(0, int(torch.min(data_len // 2)), (1,))
            else:
                frame_idx = torch.randint(0, lip.shape[-1], (1,))

            if cfg.model.which_frame_disc == "default":
                fake_frame = frame_disc(out[..., frame_idx].squeeze(-1), lip[..., 0])
                real_frame = frame_disc(lip[..., frame_idx].squeeze(-1), lip[..., 0])
            elif cfg.model.which_frame_disc == "unet":
                fake_frame_enc, fake_frame_dec = frame_disc(out[..., frame_idx].squeeze(-1), lip[..., 0])
                real_frame_enc, real_frame_dec = frame_disc(lip[..., frame_idx].squeeze(-1), lip[..., 0])

            if torch.min(data_len // 2) < lip.shape[-1]:
                seq_idx = torch.randint(0, torch.min(data_len // 2) - cfg.train.seq_length, (1,))
            else:
                seq_idx = torch.randint(0, lip.shape[-1] - cfg.train.seq_length, (1,))
            fake_seq = seq_disc(out[..., seq_idx:seq_idx + cfg.train.seq_length], feature[..., seq_idx * 2:(seq_idx + cfg.train.seq_length) * 2])
            real_seq = seq_disc(lip[..., seq_idx:seq_idx + cfg.train.seq_length], feature[..., seq_idx * 2:(seq_idx + cfg.train.seq_length) * 2])

            real_frame_idx = torch.randint(0, lip.shape[-1] - cfg.train.crop_length, (1,))
            lip_crop = lip[..., real_frame_idx:real_frame_idx + cfg.train.crop_length]
            out_crop = out[..., real_frame_idx:real_frame_idx + cfg.train.crop_length]
            feature_crop_real = feature[..., real_frame_idx * 2:(real_frame_idx + cfg.train.crop_length) * 2]
            if real_frame_idx < lip.shape[-1] // 2:
                fake_frame_idx = torch.randint(lip.shape[-1] // 2, lip.shape[-1] - cfg.train.crop_length, (1,))
            else:
                fake_frame_idx = torch.randint(0, lip.shape[-1] // 2 - cfg.train.crop_length, (1,))
            feature_crop_fake = feature[..., fake_frame_idx * 2:(fake_frame_idx + cfg.train.crop_length) * 2]
            sync_real_lip = sync_disc(lip_crop, feature_crop_real)
            sync_real_out = sync_disc(out_crop, feature_crop_real)
            sync_fake = sync_disc(lip_crop, feature_crop_fake)

        # frame disc
        if cfg.model.which_frame_disc == "default":
            frame_loss = loss_f.frame_bce_loss_d(fake_frame, real_frame)
        elif cfg.model.which_frame_disc == "unet":
            frame_loss = loss_f.frame_bce_loss_d(fake_frame_enc, real_frame_enc) + loss_f.frame_bce_loss_d(fake_frame_dec, real_frame_dec)
        wandb.log({"frame_loss_disc_val": frame_loss.item()})
        epoch_loss_frame_disc += frame_loss.item()

        # seq disc
        seq_loss = loss_f.seq_bce_loss_d(fake_seq, real_seq)
        wandb.log({"seq_loss_disc_val": seq_loss.item()})
        epoch_loss_seq_disc += seq_loss.item()

        # sync disc
        sync_loss = loss_f.sync_bce_loss_d(sync_real_lip, sync_real_out, sync_fake)
        wandb.log({"sync_loss_disc_val": sync_loss.item()})
        epoch_loss_sync_disc += sync_loss.item()
        
        # generator
        if cfg.model.which_frame_disc == "default":
            frame_loss = loss_f.frame_bce_loss_g(fake_frame)
        elif cfg.model.which_frame_disc == "unet":
            frame_loss = loss_f.frame_bce_loss_g(fake_frame_enc) + loss_f.frame_bce_loss_g(fake_frame_dec)
        seq_loss = loss_f.seq_bce_loss_g(fake_seq)
        sync_loss = loss_f.sync_bce_loss_g(sync_real_out)
        l1_loss = loss_f.l1_loss(out, lip, data_len, lip.shape[-1])
        gen_loss = cfg.train.frame_weight * frame_loss \
            + cfg.train.seq_weight * seq_loss + cfg.train.sync_weight * sync_loss + cfg.train.l1_weight * l1_loss
        epoch_loss_gen += gen_loss.item()
        wandb.log({"frame_loss_gen_val": frame_loss.item()})
        wandb.log({"seq_loss_gen_val": seq_loss.item()})
        wandb.log({"sync_loss_gen_val": sync_loss.item()})
        wandb.log({"l1_loss_gen_val": l1_loss.item()})
        wandb.log({"loss_gen_val": gen_loss.item()})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_movie(out[0], lip[0], cfg, "lip_val")
                break

        if iter_cnt > (all_iter - 1):
            check_movie(out[0], lip[0], cfg, "lip_val")
            
    epoch_loss_gen /= iter_cnt
    epoch_loss_frame_disc /= iter_cnt
    epoch_loss_seq_disc /= iter_cnt
    epoch_loss_sync_disc /= iter_cnt
    return epoch_loss_gen, epoch_loss_frame_disc, epoch_loss_seq_disc, epoch_loss_sync_disc


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    print(f"cpu_num = {os.cpu_count()}")
    print(f"gpu_num = {torch.cuda.device_count()}")
    torch.backends.cudnn.benchmark = True

    data_root = cfg.train.lip_pre_loaded_path_9696_time_only
    mean_std_path = cfg.train.lip_mean_std_path_9696_time_only
    data_root = Path(data_root).expanduser()
    mean_std_path = Path(mean_std_path).expanduser()

    # check point
    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    ckpt_path = ckpt_path / current_time
    os.makedirs(ckpt_path, exist_ok=True)

    # モデルパラメータの保存先を指定
    save_path = Path(cfg.train.save_path).expanduser()
    save_path = save_path / current_time
    os.makedirs(save_path, exist_ok=True)

    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)

    loss_f = MaskedLoss()
    train_epoch_loss_gen_list = []
    train_epoch_loss_frame_disc_list = []
    train_epoch_loss_seq_disc_list = []
    train_epoch_loss_sync_disc_list = []
    val_epoch_loss_gen_list = []
    val_epoch_loss_frame_disc_list = []
    val_epoch_loss_seq_disc_list = []
    val_epoch_loss_sync_disc_list = []
    
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        gen, frame_disc, seq_disc, sync_disc = make_model(cfg, device)

        opt_gen = torch.optim.Adam(
            params=gen.parameters(),
            lr=cfg.train.lr_gen, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay_gen    
        )
        opt_frame_disc = torch.optim.Adam(
            params=frame_disc.parameters(),
            lr=cfg.train.lr_frame_disc, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay_frame    
        )
        opt_seq_disc = torch.optim.Adam(
            params=seq_disc.parameters(),
            lr=cfg.train.lr_seq_disc, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay_seq 
        )
        opt_sync_disc = torch.optim.Adam(
            params=sync_disc.parameters(),
            lr=cfg.train.lr_sync_disc,
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay_sync
        )

        scheduler_gen = torch.optim.lr_scheduler.StepLR(
            optimizer=opt_gen,
            step_size=cfg.train.step_size,
            gamma=cfg.train.gamma_gen,
        )
        scheduler_frame_disc = torch.optim.lr_scheduler.StepLR(
            optimizer=opt_frame_disc,
            step_size=cfg.train.step_size,
            gamma=cfg.train.gamma_disc,
        )
        scheduler_seq_disc = torch.optim.lr_scheduler.StepLR(
            optimizer=opt_seq_disc,
            step_size=cfg.train.step_size,
            gamma=cfg.train.gamma_disc,
        )
        scheduler_sync_disc = torch.optim.lr_scheduler.StepLR(
            optimizer=opt_sync_disc,
            step_size=cfg.train.step_size,
            gamma=cfg.train.gamma_disc,
        )

        last_epoch = 0
        wandb.watch(gen, **cfg.wandb_conf.watch)
        wandb.watch(frame_disc, **cfg.wandb_conf.watch)
        wandb.watch(seq_disc, **cfg.wandb_conf.watch)
        wandb.watch(sync_disc, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = epoch + last_epoch
            print(f"##### {current_epoch} #####")

            # train
            epoch_loss_gen, epoch_loss_frame_disc, epoch_loss_seq_disc, epoch_loss_sync_disc = train_one_epoch(
                gen=gen,
                frame_disc=frame_disc,
                seq_disc=seq_disc,
                sync_disc=sync_disc,
                train_loader=train_loader,
                loss_f=loss_f,
                opt_gen=opt_gen,
                opt_frame_disc=opt_frame_disc,
                opt_seq_disc=opt_seq_disc,
                opt_sync_disc=opt_sync_disc,
                device=device,
                cfg=cfg,
            )
            train_epoch_loss_gen_list.append(epoch_loss_gen)
            train_epoch_loss_frame_disc_list.append(epoch_loss_frame_disc)
            train_epoch_loss_seq_disc_list.append(epoch_loss_seq_disc)
            train_epoch_loss_sync_disc_list.append(epoch_loss_sync_disc)

            # validation
            epoch_loss_gen, epoch_loss_frame_disc, epoch_loss_seq_disc, epoch_loss_sync_disc = calc_val_loss(
                gen=gen,
                frame_disc=frame_disc,
                seq_disc=seq_disc,
                sync_disc=sync_disc,
                val_loader=val_loader,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
            )
            val_epoch_loss_gen_list.append(epoch_loss_gen)
            val_epoch_loss_frame_disc_list.append(epoch_loss_frame_disc)
            val_epoch_loss_seq_disc_list.append(epoch_loss_seq_disc)
            val_epoch_loss_sync_disc_list.append(epoch_loss_sync_disc)

            scheduler_gen.step()
            scheduler_frame_disc.step()
            scheduler_seq_disc.step()
            scheduler_sync_disc.step()

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    gen=gen,
                    frame_disc=frame_disc,
                    seq_disc=seq_disc,
                    sync_disc=sync_disc,
                    opt_gen=opt_gen,
                    opt_frame_disc=opt_frame_disc,
                    opt_seq_disc=opt_seq_disc,
                    opt_sync_disc=opt_sync_disc,
                    scheduler_gen=scheduler_gen,
                    scheduler_frame_disc=scheduler_frame_disc,
                    scheduler_seq_disc=scheduler_seq_disc,
                    scheduler_sync_disc=scheduler_sync_disc,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{cfg.model.name}_{current_epoch}.ckpt"),
                )
            
            save_loss(train_epoch_loss_gen_list, val_epoch_loss_gen_list, save_path, "gen_loss")
            save_loss(train_epoch_loss_frame_disc_list, val_epoch_loss_frame_disc_list, save_path, "frame_disc_loss")
            save_loss(train_epoch_loss_seq_disc_list, val_epoch_loss_seq_disc_list, save_path, "seq_disc_loss")
            save_loss(train_epoch_loss_sync_disc_list, val_epoch_loss_sync_disc_list, save_path, "sync_disc_loss")
        
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(gen.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)

    wandb.finish()


if __name__ == "__main__":
    main()