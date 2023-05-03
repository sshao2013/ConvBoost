from torch import nn


class CNN2D_3L(nn.Module):
    """Model for human-activity-recognition."""

    def __init__(self, DB, num_classes, win_size, cnn_channel):
        super().__init__()

        # Extract features, 1D conv layers
        kernal = (5, 1)

        self.features = nn.Sequential(
            # nn.Conv2d(1, cnn_channel, kernel_size=kernal),
            nn.Conv2d(DB, cnn_channel, kernel_size=kernal),
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

        # Classify output, fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear((win_size - 28) // 4 * DB * cnn_channel, 128),
            nn.Linear((win_size - 28) // 4 * cnn_channel, 128),
            # nn.Linear(465920, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # input size: (batch_size, 1, channel, win)
        x = x.permute(0, 3, 2, 1)
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        out = self.classifier(x)

        return out
