import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_pad_mask(lengths, max_len):
    """
    口唇動画,音響特徴量に対してパディングした部分を隠すためのマスク
    """
    # この後の処理でリストになるので先にdeviceを取得しておく
    device = lengths.device

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand     
    return mask.unsqueeze(1).to(device=device)


def calc_delta(lip):
    lip = lip.to('cpu').detach().numpy().copy()
    lip_pad = 0.30*lip[0:1] + 0.59*lip[1:2] + 0.11*lip[2:3]
    lip_pad = lip_pad.astype(lip.dtype)
    lip_pad = gaussian_filter(lip_pad, (0, 0.5, 0.5, 0), mode="reflect", truncate=2)
    lip_pad = np.pad(lip_pad, ((0, 0), (0, 0), (0, 0), (1, 1)), "edge")
    lip_diff = (lip_pad[..., 2:] - lip_pad[..., :-2]) / 2
    lip_acc = lip_pad[..., 0:-2] + lip_pad[..., 2:] - 2 * lip_pad[..., 1:-1]
    lip = np.vstack((lip, lip_diff, lip_acc))
    lip = torch.from_numpy(lip)
    return lip


class MaskedLoss:
    def __init__(self):
        pass

    def calc_mask_mean(self, loss, mask):
        """
        マスクを考慮した平均を計算
        """
        loss = torch.where(mask == 0, loss, torch.tensor(0).to(device=loss.device, dtype=loss.dtype))
        loss = torch.mean(loss, dim=1)
        ones = torch.ones_like(mask).to(device=loss.device, dtype=loss.dtype)
        n_loss = torch.where(mask == 0, ones, torch.tensor(0).to(device=loss.device, dtype=loss.dtype))
        loss = torch.sum(loss) / torch.sum(n_loss)
        return loss

    def frame_bce_loss_d(self, fake, real):
        """
        discriminatorはfakeを0,realを1にするように学習
        """
        loss = F.binary_cross_entropy(fake, torch.zeros_like(fake))\
            + F.binary_cross_entropy(real, torch.ones_like(real))
        return loss
    
    def frame_bce_loss_g(self, fake):
        """
        generatorはfakeを1だとdiscriminatorに判別させるように学習
        """
        loss = F.binary_cross_entropy(fake, torch.ones_like(fake))
        return loss

    def seq_bce_loss_d(self, fake, real, data_len, max_len):
        data_len = torch.div(data_len, 2).to(dtype=torch.int)
        mask = make_pad_mask(data_len, max_len)
        eps = 1e-6
        loss = -(torch.log(real + eps) + torch.log(1 - fake + eps)) 
        loss = self.calc_mask_mean(loss, mask)
        return loss

    def seq_bce_loss_g(self, fake, data_len, max_len):
        data_len = torch.div(data_len, 2).to(dtype=torch.int)
        mask = make_pad_mask(data_len, max_len)
        eps = 1e-6
        loss = -(torch.log(fake + eps)) 
        loss = self.calc_mask_mean(loss, mask)
        return loss

    def sync_bce_loss_d(self, real_lip, real_pred, fake):
        loss = F.binary_cross_entropy(real_lip, torch.ones_like(real_lip))\
            + (F.binary_cross_entropy(fake, torch.zeros_like(fake)) + F.binary_cross_entropy(real_pred, torch.zeros_like(real_pred))) / 2
        return loss

    def sync_bce_loss_g(self, real_pred):
        loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred))
        return loss

    def l1_loss(self, pred, target, data_len, max_len):
        data_len = torch.div(data_len, 2).to(dtype=torch.int)
        mask = make_pad_mask(data_len, max_len)
        mask = mask.unsqueeze(1).unsqueeze(1)   # (B, 1, 1, 1, T)   
        loss = torch.abs(pred - target)     # (B, C, H, W, T)
        loss = self.calc_mask_mean(loss, mask)
        return loss


def main():
    loss_f = MaskedLoss()
    fake_frame = torch.rand(4, 64)
    real_frame = torch.rand_like(fake_frame)
    fake_movie = torch.rand(4, 128, 150)
    real_movie = torch.rand_like(fake_movie)
    data_len = torch.tensor([200, 200, 200, 200])

    # d
    frame_loss = loss_f.frame_bce_loss_d(fake_frame, real_frame)
    frame_loss_torch = F.binary_cross_entropy(fake_frame, torch.zeros_like(fake_frame))\
        + F.binary_cross_entropy(real_frame, torch.ones_like(real_frame))
    
    seq_loss = loss_f.seq_bce_loss_d(fake_movie, real_movie, data_len, fake_movie.shape[-1])
    seq_loss_torch = F.binary_cross_entropy(fake_movie, torch.zeros_like(fake_movie))\
        + F.binary_cross_entropy(real_movie, torch.ones_like(real_movie))
    print(frame_loss, frame_loss_torch)
    print(seq_loss, seq_loss_torch)
    # breakpoint()
    # g
    frame_loss = loss_f.frame_bce_loss_g(fake_frame)
    frame_loss_torch = F.binary_cross_entropy(fake_frame, torch.ones_like(fake_frame))

    seq_loss = loss_f.seq_bce_loss_g(fake_movie, data_len, fake_movie.shape[-1])
    seq_loss_torch = F.binary_cross_entropy(fake_movie, torch.ones_like(fake_movie))
    print(frame_loss, frame_loss_torch)
    print(seq_loss, seq_loss_torch)
    # breakpoint()

if __name__ == "__main__":
    main()