import torch.nn as nn
class CRNN(nn.Module):
    def __init__(self, input_channels, hidden_size, num_classes):
        super(CRNN, self).__init__()
        # CNN部分
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            # 输入通道数为input_channels，输出通道数为64，使用3x3的卷积核，步长为1，padding为1

            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化，使用2x2的池化窗口，步长为2

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 输入通道数为64，输出通道数为128，使用3x3的卷积核，步长为1，padding为1
            # [16, 128, 192000]

            nn.ReLU(inplace=True),  # ReLU激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化，使用2x2的池化窗口，步长为2
            # [16, 128, 96000]
        )
        # RNN部分 输入#[16,96000,128]
        self.rnn = nn.LSTM(128, hidden_size, bidirectional=True,
                           batch_first=True)  # 输入大小为128，隐藏单元数量为hidden_size，双向LSTM，批次维度在第一个维度
        # 输入张量的形状应为 (batch_size, seq_length, input_size)，其中 batch_size 是批次大小，seq_length 是序列长度，input_size 是输入特征的大小

        # 全连接层 输入为[16, 2*128]
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 输入大小为hidden_size*2，输出大小为num_classes

    def forward(self, x):
        # CNN前向传播
        x = self.cnn(x)

        batch_size, channels, height, width = x.size()

        x = x.view(batch_size, channels, height * width)

        x = x.permute(0, 2, 1)
        output, m = self.rnn(x)
        output = self.fc(output[:, -1, :])
        return output.view(batch_size, -1)