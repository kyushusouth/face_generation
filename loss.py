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


class MaskedLoss:
    def __init__(self):
        self.eps = 1.0e-6
        pass

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

    def seq_bce_loss_d(self, fake, real):
        loss = F.binary_cross_entropy(fake, torch.zeros_like(fake))\
            + F.binary_cross_entropy(real, torch.ones_like(real))
        return loss

    def seq_bce_loss_g(self, fake):
        loss = F.binary_cross_entropy(fake, torch.ones_like(fake))
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
        loss = torch.where(mask == 0, loss, torch.zeros_like(loss))
        loss = torch.mean(loss, dim=1)  # (B, H, W, T)
        
        mask = mask.squeeze(1)  # (B, 1, 1, T)
        n_loss = torch.where(mask == 0, torch.ones_like(mask).to(torch.float32), torch.zeros_like(mask).to(torch.float32))
        n_loss = n_loss.expand_as(loss)     # (B, H, W, T)
        loss = torch.sum(loss) / torch.sum(n_loss)
        return loss


def main():
    loss_f = MaskedLoss()
    fake_frame = torch.rand(4, 64)
    real_frame = torch.rand_like(fake_frame)
    fake_movie = torch.rand(4, 128, 150)
    real_movie = torch.rand_like(fake_movie)
    data_len = torch.tensor([30, 40, 20])

    hw = 4
    lip = torch.rand(3, 3, hw, hw, 15)
    pred = torch.rand(3, 3, hw, hw, 15)
    loss = loss_f.l1_loss(pred, lip, data_len, max_len=lip.shape[-1])
    print(loss)


if __name__ == "__main__":
    main()