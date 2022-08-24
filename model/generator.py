"""
generatorは最後の活性関数をtanhにすることに注意
画像は基本[0, 255]の決められた範囲の値しか取らないので、標準化でなく正規化が用いられることが多いみたい
それに倣い、[-1, 1]に正規化するので出力の範囲も合わせる
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .audio_encoder import MelEncoder
except:
    from audio_encoder import MelEncoder


class Generator(nn.Module):
    def __init__(self, in_channels, img_cond_channels, feat_channels, feat_cond_channels, noise_channels, dropout, tc_ksize):
        super().__init__()
        assert tc_ksize % 2 == 0
        self.noise_channels = noise_channels
        in_cs = [in_channels, 64, 128, 256]
        out_cs = [64, 128, 256, 512]

        if tc_ksize == 2:
            padding = 0
        elif tc_ksize == 4:
            padding = 1

        self.audio_enc = MelEncoder(in_channels=feat_channels, out_channels=feat_cond_channels)
        self.noise_rnn = nn.GRU(noise_channels, noise_channels, num_layers=1, batch_first=True, bidirectional=True)
        self.noise_fc = nn.Linear(noise_channels * 2, noise_channels)

        self.enc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
            ) for in_c, out_c in zip(in_cs, out_cs)
        ])
        self.enc_last_layer = nn.Sequential(
            nn.Conv2d(out_cs[-1], img_cond_channels, kernel_size=3),
            nn.ReLU(),
        )

        self.dec_first_layer = nn.Sequential(
            nn.ConvTranspose3d(img_cond_channels + self.audio_enc.out_channels + noise_channels, out_cs[-1], kernel_size=(3, 3, 1)),
            nn.BatchNorm3d(out_cs[-1]),
            nn.ReLU(),
        )
        self.dec_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(out_c * 2, out_c * 2, kernel_size=(3, 3, 1), padding=(1, 1, 0)),   
                nn.ConvTranspose3d(out_c * 2, in_c, kernel_size=(tc_ksize, tc_ksize, 1), stride=(2, 2, 1), padding=(padding, padding, 0)),
                nn.BatchNorm3d(in_c),
                nn.ReLU(),
                nn.Dropout(dropout),
            ) for in_c, out_c, in zip(list(reversed(in_cs))[:-1], list(reversed(out_cs))[:-1])
        ])
        self.dec_last_layer = nn.Sequential(
            nn.Conv3d(out_cs[0] * 2, out_cs[0] * 2, kernel_size=(3, 3, 1), padding=(1, 1, 0)),
            nn.ConvTranspose3d(out_cs[0] * 2, in_cs[0], kernel_size=(tc_ksize, tc_ksize, 1), stride=(2, 2, 1), padding=(padding, padding, 0)),
            nn.BatchNorm3d(in_cs[0]),
            nn.Tanh(),  # 出力が[-1, 1]になるようにする(事前に画像を[-1, 1]にしておく)
        )

    def forward(self, lip, feature):
        """
        lip : (B, C, H, W)
        feature : (B, C, T)
        out : (B, C, H, W, T)
        """
        enc_out = lip
        fmaps = []
        for layer in self.enc_layers:
            enc_out = layer(enc_out)
            fmaps.append(enc_out)

        # 画像から得られる見た目についての特徴表現
        lip_rep = self.enc_last_layer(enc_out)  # (B, C, 1, 1)

        # 音響特徴量から得られる特徴表現
        feat_rep = self.audio_enc(feature)  # (B, C, T)

        # ノイズを生成することで音声によらない特徴を表現できるようにする(見た目)
        noise = torch.normal(mean=0, std=0.6, size=(feat_rep.shape[0], feat_rep.shape[-1], self.noise_channels)).to(device=lip.device, dtype=lip.dtype)    # (B, T, C)
        noise_rep, _ = self.noise_rnn(noise)    
        noise_rep = self.noise_fc(noise_rep)
        noise_rep = noise_rep.permute(0, 2, 1)  # (B, C, T)

        lip_rep = lip_rep.unsqueeze(-1)     # (B, C, 1, 1, 1)
        lip_rep = lip_rep.expand(-1, -1, -1, -1, feat_rep.shape[-1])  # (B, C, 1, 1, T)
        feat_rep = feat_rep.unsqueeze(2).unsqueeze(2)   # (B, C, 1, 1, T)
        noise_rep = noise_rep.unsqueeze(2).unsqueeze(2)   # (B, C, 1, 1, T)
        rep = torch.cat([lip_rep, feat_rep, noise_rep], dim=1)

        # 音声と画像から発話内容に対応した動画を合成
        out = self.dec_first_layer(rep)     # (B, C, 3, 3, T)
        for layer, fmap in zip(self.dec_layers, reversed(fmaps)):
            fmap = fmap.unsqueeze(-1).expand(-1, -1, -1, -1, out.shape[-1])
            out = torch.cat([out, fmap], dim=1)
            out = layer(out)
        
        out = torch.cat([out, fmaps[0].unsqueeze(-1).expand(-1, -1, -1, -1, out.shape[-1])], dim=1)
        out = self.dec_last_layer(out)
        return out


def main():
    net = Generator(3, 50, 80, 256, 10, 0, 4)
    lip = torch.rand(1, 3, 48, 48)
    feature = torch.rand(1, 80, 300)
    out = net(lip, feature)
    print(f"out = {out.shape}")
    breakpoint()


    trans_conv = nn.ModuleList([
        nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
        nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
        nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
        nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1),
    ])
    lip = torch.rand(1, 3, 3, 3)
    out = lip
    for layer in trans_conv:
        out = layer(out)
        print(out.shape)
    breakpoint()



if __name__ == "__main__":
    main()