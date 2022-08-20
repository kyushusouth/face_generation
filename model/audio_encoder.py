import torch
import torch.nn as nn
import torch.nn.functional as F


class MelEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        in_cs = [in_channels, 128, 256, 256, 512]
        out_cs = [128, 256, 256, 512, out_channels]
        stride = [1, 1, 2, 1, 1]
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=5, stride=s, padding=2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(), 
            ) for in_c, out_c, s in zip(in_cs, out_cs, stride)
        ])
        self.rnn = nn.GRU(out_channels, out_channels, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(out_channels * 2, out_channels)

    def forward(self, x):
        """
        x : (B, C, T)
        out : (B, C, T)
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        
        out = out.permute(0, 2, 1)  # (B, T, C)
        out, _ = self.rnn(out)  
        out = self.fc(out)
        out = out.permute(0, 2, 1)  # (B, C, T)

        return out


def main():
    net = MelEncoder()
    feature = torch.rand(1, 80, 300)
    out = net(feature)
    breakpoint()


if __name__ == "__main__":
    main()
