import torch.nn as nn


class ConvLSTM(nn.Module):
    """Model for human-activity-recognition."""

    def __init__(self, input_channel, num_classes, cnn_channel):
        super().__init__()
        self.n_layers = 2
        self.num_classes = num_classes
        self.n_hidden = 128

        kernal = (5, 1)
        # self.features = nn.Sequential(
        #     # nn.Conv2d(1, cnn_channel, kernel_size=kernal),
        #     nn.Conv2d(input_channel, cnn_channel, kernel_size=kernal),
        #     nn.MaxPool2d((2, 1)),
        #     nn.Conv2d(cnn_channel, cnn_channel, kernel_size=kernal),
        #     nn.MaxPool2d((2, 1)),
        #     nn.Conv2d(cnn_channel, cnn_channel, kernel_size=kernal),
        # )

        self.features = nn.Sequential(
            nn.Conv2d(input_channel, cnn_channel, kernel_size=kernal),
            nn.GroupNorm(4, cnn_channel),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn_channel, cnn_channel, kernel_size=kernal),
            nn.GroupNorm(4, cnn_channel),
            nn.MaxPool2d((2, 1)),
            nn.ReLU(),
            nn.Conv2d(cnn_channel, cnn_channel, kernel_size=kernal),
            nn.GroupNorm(4, cnn_channel),
            nn.ReLU(),
            # nn.AdaptiveMaxPool2d((4, input_channel))
        )

        self.lstm1 = nn.LSTM(cnn_channel, hidden_size=self.n_hidden, num_layers=self.n_layers)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = x.permute(0, 3, 2, 1)
        x = self.features(x)

        x = x.permute(2, 0, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        # x, _ = self.lstm1(x)
        # x = self.dropout(x)
        # out = self.fc(x[:, -1])

        x = self.dropout(x)
        x, _ = self.lstm1(x)
        x = x[-1, :, :]
        # x = x.view(x.shape[0], -1, 128)
        out = self.fc(x)

        return out
