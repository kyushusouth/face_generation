import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameDiscriminator(nn.Module):
    def __init__(self, in_channels, dropout):
        super().__init__()
        in_cs = [in_channels, 64, 128, 256]
        out_cs = [64, 128, 256, 512]

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs, out_cs)
        ])
        self.last_layer = nn.Sequential(
            nn.Conv2d(out_cs[-1], out_cs[0], kernel_size=3),
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
        return out.squeeze(-1).squeeze(-1)


class SequenceDiscriminator(nn.Module):
    def __init__(self, in_channels, feat_channels, dropout):
        super().__init__()
        in_cs_lip = [in_channels, 64, 128, 256]
        out_cs_lip = [64, 128, 256, 512]
        self.lip_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0)),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs_lip, out_cs_lip)
        ])
        self.lip_last_layer = nn.Sequential(
            nn.Conv3d(out_cs_lip[-1], out_cs_lip[0], kernel_size=(3, 3, 1)),
            nn.BatchNorm3d(out_cs_lip[0]),
            nn.LeakyReLU(0.2),
        )

        in_cs_feat = [feat_channels, 128, 256, 256, 512]
        out_cs_feat = [128, 256, 256, 512, out_cs_lip[0]]
        stride = [1, 1, 2, 1, 1]
        self.feat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.2), 
                nn.Dropout(dropout),
            ) for in_c, out_c, s in zip(in_cs_feat, out_cs_feat, stride)
        ])
        self.lip_rnn = nn.GRU(out_cs_lip[0], out_cs_lip[0], num_layers=1, batch_first=True, bidirectional=True)
        self.feat_rnn = nn.GRU(out_cs_lip[0], out_cs_lip[0], num_layers=1, batch_first=True, bidirectional=True)
    
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
        out = out.permute(0, 2, 1)  # (B, C, T)
        out = torch.sigmoid(out)
        return out


class SyncDiscriminator(nn.Module):
    def __init__(self, in_channels, feat_channels, crop_length, dropout):
        super().__init__()
        in_cs_lip = [in_channels, 64, 128, 256]
        out_cs_lip = [64, 128, 256, 512]
        self.lip_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=1),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
            ) for in_c, out_c in zip(in_cs_lip, out_cs_lip)
        ])
        self.lip_last_layer = nn.Sequential(
            nn.Conv3d(out_cs_lip[-1], out_cs_lip[-2], kernel_size=(3, 3, 1)),
            nn.BatchNorm3d(out_cs_lip[-2]),
            nn.LeakyReLU(0.2),
        )

        in_cs_feat = [feat_channels, 128, 256, 256]
        out_cs_feat = [128, 256, 256, 512]
        stride = [1, 1, 2, 1]
        self.feat_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2),
                nn.BatchNorm1d(out_c),
                nn.LeakyReLU(0.2), 
                nn.Dropout(dropout),
            ) for in_c, out_c, s in zip(in_cs_feat, out_cs_feat, stride)
        ])
        self.feat_last_layer = nn.Sequential(
            nn.Conv1d(512, out_cs_lip[-2], kernel_size=5, padding=2),
            nn.BatchNorm1d(out_cs_lip[-2]),
            nn.LeakyReLU(0.2),
        )

        self.out_fc = nn.Linear(crop_length, crop_length)

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
    net = FrameDiscriminator(in_channels=6)
    lip = torch.rand(1, 3, 48, 48, 150)
    frame_idx = torch.randint(0, lip.shape[0], (1,))
    out = net(lip[..., frame_idx].squeeze(-1), lip[..., 0])
    breakpoint()

    net = SequenceDiscriminator(in_channels=3)
    lip = torch.rand(1, 3, 48, 48, 150)
    feature = torch.rand(1, 80, 300)
    out = net(lip, feature)
    breakpoint()

    net = SyncDiscriminator(in_channels=3)
    lip = torch.rand(1, 3, 48, 48, 50)
    feature = torch.rand(1, 80, 100)
    lip = torch.ones_like(lip)
    distance = net(lip, feature)
    breakpoint()


if __name__ == "__main__":
    main()
