import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class FrameDiscriminator(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()
        in_cs = [in_channels, 64, 128, 256]
        out_cs = [64, 128, 256, 512]

        self.layers = nn.ModuleList([
            nn.Sequential(
                # spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs, out_cs)
        ])
        self.last_layer = nn.Sequential(
            nn.Conv2d(out_cs[-1], out_cs[0], kernel_size=3),
            nn.Conv2d(out_cs[0], 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, lip, lip_cond):
        """
        ランダムに選択した1フレームの画像に対する判別を行う
        lip : (B, C, H, W)      ランダムに選択したフレーム
        lip_cond : (B, C, H, W)     合成に使用したフレーム
        out : (B, C)
        """
        lip = torch.cat([lip, lip_cond], dim=1)
        out = lip
        for layer in self.layers:
            out = layer(out)
        out = self.last_layer(out)
        return out


class FrameDiscriminatorUNet(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()
        in_cs = [in_channels, 64, 128, 256]
        out_cs = [64, 128, 256, 512]

        self.enc_layers = nn.ModuleList([
            nn.Sequential(
                # spectral_norm(nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)),
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs, out_cs)
        ])
        self.enc_last_layer = nn.Conv2d(out_cs[-1], out_cs[0], kernel_size=3)
        self.enc_out_layer = nn.Conv2d(out_cs[0], 1, kernel_size=1)

        self.dec_first_layer = nn.Sequential(
            # spectral_norm(nn.ConvTranspose2d(out_cs[0], out_cs[-1], kernel_size=3)),
            nn.ConvTranspose2d(out_cs[0], out_cs[-1], kernel_size=3),
            nn.BatchNorm2d(out_cs[-1]),
            nn.LeakyReLU(0.2),
        )
        self.dec_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_c * 2, out_c * 2, kernel_size=3, padding=1),
                # spectral_norm(nn.ConvTranspose2d(out_c * 2, in_c, kernel_size=4, stride=2, padding=1)),
                nn.ConvTranspose2d(out_c * 2, in_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(in_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(list(reversed(in_cs))[:-1], list(reversed(out_cs))[:-1])
        ])
        self.dec_last_layer = nn.Sequential(
            nn.Conv2d(out_cs[0] * 2, out_cs[0] * 2, kernel_size=3, padding=1),
            # spectral_norm(nn.ConvTranspose2d(out_cs[0] * 2, 3, kernel_size=4, stride=2, padding=1)),
            nn.ConvTranspose2d(out_cs[0] * 2, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
        )

    def forward(self, lip, lip_cond):
        """
        ランダムに選択した1フレームの画像に対する判別を行う
        lip : (B, C, H, W)      ランダムに選択したフレーム
        lip_cond : (B, C, H, W)     合成に使用したフレーム
        enc_out : (B, C)
        dec_out : (B, C, H, W)
        """
        lip = torch.cat([lip, lip_cond], dim=1)
        enc_out = lip
        fmaps = []
        for layer in self.enc_layers:
            enc_out = layer(enc_out)
            fmaps.append(enc_out)
        enc_out = self.enc_last_layer(enc_out)

        dec_out = self.dec_first_layer(enc_out)
        for layer, fmap in zip(self.dec_layers, list(reversed(fmaps))):
            dec_out = torch.cat([dec_out, fmap], dim=1)
            dec_out = layer(dec_out)
        dec_out = torch.cat([dec_out, fmaps[0]], dim=1)
        dec_out = self.dec_last_layer(dec_out)

        enc_out = self.enc_out_layer(enc_out)
        enc_out = torch.sigmoid(enc_out)
        dec_out = torch.sigmoid(dec_out)
        return enc_out, dec_out


class SequenceDiscriminator(nn.Module):
    def __init__(self, in_channels, feat_channels, dropout, analysis_len):
        super().__init__()
        in_cs_lip = [in_channels, 64, 128, 256]
        out_cs_lip = [64, 128, 256, 512]
        self.lip_layers = nn.ModuleList([
            nn.Sequential(
                # spectral_norm(nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))),
                nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs_lip, out_cs_lip)
        ])
        self.lip_last_layer = nn.Sequential(
            # spectral_norm(nn.Conv3d(out_cs_lip[-1], out_cs_lip[0], kernel_size=(3, 3, 1))),
            nn.Conv3d(out_cs_lip[-1], out_cs_lip[0], kernel_size=(3, 3, 1)),
            nn.BatchNorm3d(out_cs_lip[0]),
            nn.LeakyReLU(0.2),
        )

        in_cs_feat = [feat_channels, 128, 256, 256, 512]
        out_cs_feat = [128, 256, 256, 512, out_cs_lip[0]]
        stride = [1, 1, 2, 1, 1]
        self.feat_layers = nn.ModuleList([
            nn.Sequential(
                # spectral_norm(nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2)),
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.2), 
                nn.Dropout(dropout),
            ) for in_c, out_c, s in zip(in_cs_feat, out_cs_feat, stride)
        ])
        self.lip_rnn = nn.GRU(out_cs_lip[0], out_cs_lip[0], num_layers=1, batch_first=True, bidirectional=True)
        self.feat_rnn = nn.GRU(out_cs_lip[0], out_cs_lip[0], num_layers=1, batch_first=True, bidirectional=True)
        self.out_layers= nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_cs_lip[0] * 2 * analysis_len, 1)
        )
    
    def forward(self, lip, feature):
        """
        動画と音響特徴量を利用し,系列全体から判定する
        lip : (B, C, H, W, T)
        feature : (B, T, C)
        """
        lip_rep = lip
        for layer in self.lip_layers:
            lip_rep = layer(lip_rep)
        lip_rep = self.lip_last_layer(lip_rep)
        lip_rep = lip_rep.squeeze(2).squeeze(2)     # (B, C, T)
        
        feat_rep = feature
        for layer in self.feat_layers:
            feat_rep = layer(feat_rep)
        
        lip_rep = lip_rep.permute(0, 2, 1)  # (B, T, C)
        feat_rep = feat_rep.permute(0, 2, 1)    #(B, T, C)
        lip_rep, _ = self.lip_rnn(lip_rep)
        feat_rep, _ = self.feat_rnn(feat_rep)
        out = lip_rep + feat_rep
        out = self.out_layers(out)
        out = torch.sigmoid(out)
        return out


class SyncDiscriminator(nn.Module):
    def __init__(self, in_channels, feat_channels, crop_length, dropout):
        super().__init__()
        in_cs_lip = [in_channels, 64, 128, 256]
        out_cs_lip = [64, 128, 256, 512]
        self.lip_layers = nn.ModuleList([
            nn.Sequential(
                # spectral_norm(nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=1)),
                nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=1),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs_lip, out_cs_lip)
        ])
        self.lip_last_layer = nn.Sequential(
            # spectral_norm(nn.Conv3d(out_cs_lip[-1], out_cs_lip[-2], kernel_size=(3, 3, 1))),
            nn.Conv3d(out_cs_lip[-1], out_cs_lip[-2], kernel_size=(3, 3, 1)),
            nn.BatchNorm3d(out_cs_lip[-2]),
            nn.LeakyReLU(0.2),
        )

        in_cs_feat = [feat_channels, 128, 256, 256]
        out_cs_feat = [128, 256, 256, 512]
        stride = [1, 1, 2, 1]
        self.feat_layers = nn.ModuleList([
            nn.Sequential(
                # spectral_norm(nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2)),
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.2), 
                nn.Dropout(dropout),
            ) for in_c, out_c, s in zip(in_cs_feat, out_cs_feat, stride)
        ])
        self.feat_last_layer = nn.Sequential(
            # spectral_norm(nn.Conv1d(512, out_cs_lip[-2], kernel_size=5, padding=2)),
            nn.Conv1d(512, out_cs_lip[-2], kernel_size=5, padding=2),
            nn.BatchNorm1d(out_cs_lip[-2]),
            nn.LeakyReLU(0.2),
        )

        self.out_fc = nn.Linear(crop_length, 1)

    def forward(self, lip, feature):
        """
        適当な長さの音声と口唇動画のペアから,それらが同期しているかどうか判別
        lip : (B, C, H, W, T)
        feature : (B, C, T)
        """
        lip_rep = lip
        for layer in self.lip_layers:
            lip_rep = layer(lip_rep)
        lip_rep = self.lip_last_layer(lip_rep)
        lip_rep = lip_rep.squeeze(2).squeeze(2)

        feat_rep = feature
        for layer in self.feat_layers:
            feat_rep = layer(feat_rep)
        feat_rep = self.feat_last_layer(feat_rep)

        distance = torch.sum((lip_rep - feat_rep)**2, dim=1).sqrt()
        distance = self.out_fc(distance)
        distance = torch.sigmoid(distance)
        return distance


def main():
    dropout = 0
    net = FrameDiscriminator(in_channels=6, dropout=dropout)
    lip = torch.rand(1, 3, 48, 48, 150)
    frame_idx = torch.randint(0, lip.shape[0], (1,))
    out = net(lip[..., frame_idx].squeeze(-1), lip[..., 0])
    breakpoint()

    net = FrameDiscriminatorUNet(in_channels=6, dropout=dropout)
    lip = torch.rand(1, 3, 48, 48, 150)
    frame_idx = torch.randint(0, lip.shape[0], (1,))
    enc_out, dec_out = net(lip[..., frame_idx].squeeze(-1), lip[..., 0])
    breakpoint()

    net = SequenceDiscriminator(in_channels=3, feat_channels=80, dropout=dropout, analysis_len=150)
    lip = torch.rand(1, 3, 48, 48, 150)
    feature = torch.rand(1, 80, 300)
    out = net(lip, feature)
    breakpoint()

    net = SyncDiscriminator(in_channels=3, feat_channels=80, crop_length=50, dropout=dropout)
    lip = torch.rand(1, 3, 48, 48, 50)
    feature = torch.rand(1, 80, 100)
    lip = torch.ones_like(lip)
    out = net(lip, feature)
    breakpoint()


if __name__ == "__main__":
    main()
