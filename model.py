import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),  # (B,64,H,W)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # (B,64,H/2,W/2)

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # (B,128,H/4,W/4)

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
        )

        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=nh,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.linear = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        # x: (B, C, H, W)
        conv = self.cnn(x)   # -> (B, C, H', W')

        b, c, h, w = conv.size()

        # collapse height dimension
        if conv.size(2) > 1:
            conv = F.adaptive_avg_pool2d(conv, (1, conv.shape[3])) 
            B, C, H, W = conv.shape
            # conv = conv.view(B, C * H, W)  # → (B, C*H′, W′)
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)  # (B, W', C)

        rnn_out, _ = self.rnn(conv)  # (B, W', 2*hidden)
        output = self.linear(rnn_out)  # (B, W', nclass)

        return output.permute(1, 0, 2)  # (T, B, nclass)
