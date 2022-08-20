"""
画像は[-1, 1]の範囲に正規化する
"""

import os
import sys

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torch.utils.data import Dataset


def get_datasets(data_root, name):    
    """
    npzファイルのパス取得
    """
    items = []
    for curdir, dir, files in os.walk(data_root):
        for file in files:
            if file.endswith(".npz"):
                # mspecかworldかの分岐
                if f"{name}" in Path(file).stem:
                    data_path = os.path.join(curdir, file)
                    if os.path.isfile(data_path):
                        items.append(data_path)
    return items


def load_mean_std(mean_std_path, name, test):
    """
    一応複数話者の場合は全話者の平均にできるようにやってみました
    """
    each_lip_mean = []
    each_lip_std = []
    each_feat_mean = []
    each_feat_std = []
    each_feat_add_mean = []
    each_feat_add_std = []

    # 話者ごとにリスト
    for curdir, dirs, files in os.walk(mean_std_path):
        for file in files:
            if file.endswith('.npz'):
                if f"{name}" in Path(file).stem:
                    if test:
                        if f"test" in Path(file).stem:
                            npz_key = np.load(os.path.join(curdir, file))
                            each_lip_mean.append(torch.from_numpy(npz_key['lip_mean']))
                            each_lip_std.append(torch.from_numpy(npz_key['lip_std']))
                            each_feat_mean.append(torch.from_numpy(npz_key['feat_mean']))
                            each_feat_std.append(torch.from_numpy(npz_key['feat_std']))
                            each_feat_add_mean.append(torch.from_numpy(npz_key['feat_add_mean']))
                            each_feat_add_std.append(torch.from_numpy(npz_key['feat_add_std']))
                    else:
                        if f"train" in Path(file).stem:
                            npz_key = np.load(os.path.join(curdir, file))
                            each_lip_mean.append(torch.from_numpy(npz_key['lip_mean']))
                            each_lip_std.append(torch.from_numpy(npz_key['lip_std']))
                            each_feat_mean.append(torch.from_numpy(npz_key['feat_mean']))
                            each_feat_std.append(torch.from_numpy(npz_key['feat_std']))
                            each_feat_add_mean.append(torch.from_numpy(npz_key['feat_add_mean']))
                            each_feat_add_std.append(torch.from_numpy(npz_key['feat_add_std']))

    # 話者人数で割って平均
    lip_mean = sum(each_lip_mean) / len(each_lip_mean)
    lip_std = sum(each_lip_std) / len(each_lip_std)
    feat_mean = sum(each_feat_mean) / len(each_feat_mean)
    feat_std = sum(each_feat_std) / len(each_feat_std)
    feat_add_mean = sum(each_feat_add_mean) / len(each_feat_add_mean)
    feat_add_std = sum(each_feat_add_std) / len(each_feat_add_std)

    return lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std


class FaceGenDataset(Dataset):
    def __init__(self, data_path, mean_std_path, transform, cfg, test):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.test = test
        
        self.lip_mean, self.lip_std, self.feat_mean, self.feat_std, self.feat_add_mean, self.feat_add_std = load_mean_std(mean_std_path, cfg.model.name, test)

        print(f"n = {self.__len__()}")
    
    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        data_path = Path(self.data_path[index])
        speaker = data_path.parents[0].name
        label = data_path.stem

        npz_key = np.load(str(data_path))
        wav = torch.from_numpy(npz_key['wav'])
        lip = torch.from_numpy(npz_key['lip'])
        feature = torch.from_numpy(npz_key['feature'])
        feat_add = torch.from_numpy(npz_key['feat_add'])
        upsample = torch.from_numpy(npz_key['upsample'])
        data_len = torch.from_numpy(npz_key['data_len'])

        lip, feature, feat_add, data_len = self.transform(
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            upsample=upsample,
            data_len=data_len, 
            lip_mean=self.lip_mean, 
            lip_std=self.lip_std, 
            feat_mean=self.feat_mean, 
            feat_std=self.feat_std, 
            feat_add_mean=self.feat_add_mean, 
            feat_add_std=self.feat_add_std, 
        )
        return wav, lip, feature, feat_add, upsample, data_len, speaker, label


class FaceGenTransform:
    def __init__(self, cfg, train_val_test=None):
        self.color_jitter = T.ColorJitter(brightness=[0.5, 1.5], contrast=0, saturation=1, hue=0.2)
        self.blur = T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 5)) 
        self.horizontal_flip = T.RandomHorizontalFlip(p=0.5)
        self.rotation = T.RandomRotation(degrees=(0, 10))
        self.pad = T.RandomCrop(size=(48, 48), padding=4)
        self.cfg = cfg
        self.train_val_test = train_val_test

    def normalization(self, feature, feat_add, feat_mean, feat_std, feat_add_mean, feat_add_std):
        """
        標準化
        feature, feat_add : (C, T)
        """
        feat_mean = feat_mean.unsqueeze(-1)
        feat_std = feat_std.unsqueeze(-1)
        feat_add_mean = feat_add_mean.unsqueeze(-1)
        feat_add_std = feat_add_std.unsqueeze(-1)

        feature = (feature - feat_mean) / feat_std
        feat_add = (feat_add - feat_add_mean) / feat_add_std

        return feature, feat_add

    def time_augment(self, lip, feature, feat_add, upsample, data_len):
        """
        再生速度を変更する
        lip : (C, H, W, T)
        feature, feat_add : (C, T)
        """
        # 変更する割合を決定
        rate = torch.randint(100 - self.cfg.train.time_augment_rate, 100 + self.cfg.train.time_augment_rate, (1,)) / 100
        T = feature.shape[-1]
        T_l = lip.shape[-1]

        # 口唇動画から取得するフレームを決定
        idx = torch.linspace(0, 1, int(T * rate) // upsample * upsample)
        idx_l = (idx[::upsample] * (T_l-1)).to(torch.int)
        
        # 重複したフレームを取得する、あるいはフレームを間引くことで再生速度を変更
        new_lip = []
        for i in idx_l:
            new_lip.append(lip[..., i.item()])
        lip = torch.stack(new_lip, dim=-1)

        # 音響特徴量を補完によって動画に合わせる
        # 時間周波数領域でリサンプリングを行うことで、ピッチが変わらないようにしています
        feature = feature.unsqueeze(0)      # (1, C, T)
        feat_add = feat_add.unsqueeze(0)    # (1, C, T)
        feature = F.interpolate(feature, scale_factor=rate, mode="nearest", recompute_scale_factor=False).squeeze(0)    # (C, T)
        feat_add = F.interpolate(feat_add, scale_factor=rate, mode="nearest", recompute_scale_factor=False).squeeze(0)  # (C, T)
        
        # データの長さが変わったので、data_lenを更新して系列長を揃える
        data_len = torch.tensor(min(int(lip.shape[-1] * upsample), feature.shape[-1])).to(torch.int)
        lip = lip[..., :data_len // upsample]
        feature = feature[..., :data_len]
        feat_add = feat_add[..., :data_len]
        assert lip.shape[-1] == feature.shape[-1] // upsample
        return lip, feature, feat_add, data_len

    def __call__(self, lip, feature, feat_add, upsample, data_len, lip_mean, lip_std, feat_mean, feat_std, feat_add_mean, feat_add_std):
        """
        lip : (C, H, W, T)
        feature, feat_add : (T, C)
        """
        feature = feature.permute(-1, 0)    # (C, T)
        feat_add = feat_add.permute(-1, 0)  # (C, T)

        # data augmentation
        if self.train_val_test == "train":
            if self.cfg.train.use_time_augment:
                lip, feature, feat_add, data_len = self.time_augment(lip, feature, feat_add, upsample, data_len)

        # 標準化
        feature, feat_add = self.normalization(
            feature, feat_add, feat_mean, feat_std, feat_add_mean, feat_add_std
        )

        # 口唇動画の正規化
        lip = (lip - 127.5) / 127.5     # [-1, 1]
        
        return lip.to(torch.float32), feature.to(torch.float32), feat_add.to(torch.float32), data_len            


def collate_time_adjust(batch, cfg):
    """
    フレーム数の調整を行う
    """
    wav, lip, feature, feat_add, upsample, data_len, speaker, label = list(zip(*batch))

    lip_adjusted = []
    feature_adjusted = []
    feat_add_adjusted = []

    # configで指定した範囲でフレーム数を決定
    lip_len = torch.randint(cfg.model.lip_min_frame, cfg.model.lip_max_frame, (1,)).item()
    upsample_scale = upsample[0].item()
    feature_len = int(lip_len * upsample_scale)

    for l, f, f_add, d_len in zip(lip, feature, feat_add, data_len):
        # 揃えるlenよりも短い時
        if d_len <= feature_len:
            l_padded = torch.zeros(l.shape[0], l.shape[1], l.shape[2], lip_len)
            f_padded = torch.zeros(f.shape[0], feature_len)
            f_add_padded = torch.zeros(f_add.shape[0], feature_len)

            for t in range(l.shape[-1]):
                l_padded[..., t] = l[..., t]
            
            for t in range(f.shape[-1]):
                f_padded[:, t] = f[:, t]
                f_add_padded[:, t] = f_add[:, t]

            l = l_padded
            f = f_padded
            f_add = f_add_padded

        # 揃えるlenよりも長い時
        else:
            lip_start_frame = torch.randint(0, l.shape[-1] - lip_len, (1,)).item()
            feature_start_frame = int(lip_start_frame * upsample_scale)
            l = l[..., lip_start_frame:lip_start_frame + lip_len]
            f = f[:, feature_start_frame:feature_start_frame + feature_len]
            f_add = f_add[:, feature_start_frame:feature_start_frame + feature_len]

        assert l.shape[-1] == lip_len
        assert f.shape[-1] == feature_len
        assert f_add.shape[-1] == feature_len

        lip_adjusted.append(l)
        feature_adjusted.append(f)
        feat_add_adjusted.append(f_add)

    lip = torch.stack(lip_adjusted)
    feature = torch.stack(feature_adjusted)
    feat_add = torch.stack(feat_add_adjusted)
    data_len = torch.stack(data_len)

    return lip, feature, feat_add, upsample, data_len, speaker, label