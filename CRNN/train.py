from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn
import torch.optim as optim
import torch
import librosa
from crnn_model import CRNN
import numpy as np
from data_processing import train_loader
from data_processing import val_loader

#数据增强
def add_noise(wav, noise_factor=0.05):
    # 添加高斯噪声
    noise = torch.randn_like(wav)
    return wav + noise_factor * noise

def time_shift(wav, shift_factor=0.2):
    # 随机平移音频
    shift = int(wav.shape[-1] * shift_factor)
    return torch.roll(wav, shifts=shift, dims=-1)

    # 对音频进行数据增强
def data_augmentation(wav):
    # 添加高斯噪声
    wav = add_noise(wav)
    # 随机平移音频
    wav = time_shift(wav)
    return wav

input_size = 1  # 输入通道数
hidden_size = 128  # 隐藏层大小
num_classes = 5  # 类别数目

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建 CRNN 模型
model = CRNN(input_size, hidden_size, num_classes)
model.to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()# BCE损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-6)

def train(model, criterion, train_loader, val_loader, optimizer, epoch, device):
    model.train()  # 训练模式
    train_loss = 0
    total_samples = 0
    for batch_idx, (_, data, target, _, _) in enumerate(train_loader):
        data = data_augmentation(data)
        data = np.array(data)

        # 计算mel数值
        mel_spect = librosa.feature.melspectrogram(y=data, sr=48000, n_fft=2048, hop_length=1024)
        data = librosa.power_to_db(mel_spect, ref=np.max)  # 计算mel值
        data = torch.from_numpy(data)

        data = torch.unsqueeze(data, 1)  # 添加通道维度 [16，1, 192000]   批次 通道 宽度

        data, target = data.to(device), target.to(device)

        # bias 是float类型  加噪处理后 会变成double类型 所有要转
        data = data.float()
        optimizer.zero_grad()
        output = model(data)  # 训练

        output = torch.unsqueeze(output, 2)

        min_val = torch.min(output)
        max_val = torch.max(output)
        # 对output数据进行归一化
        output = torch.div(torch.sub(output, min_val), (max_val - min_val))

        min_val = torch.min(target)
        max_val = torch.max(target)
        # 对target数据进行归一化
        target = torch.div(torch.sub(target, min_val), (max_val - min_val))

        loss = criterion(output, target)  # 计算loss

        loss.backward()  # 反向传播
        optimizer.step()  # 优化

        batch_size = data.size(0)
        train_loss += loss.item() * batch_size  # 累加该批次的损失值乘以批次大小
        total_samples += batch_size  # 更新总样本数
        train_loss = train_loss / total_samples  # 计算整个训练集的平均损失
    print(f"Train Loss: {train_loss:.4f}")
    # ********************************************************************************************************#
    model.eval()  # 评估模式
    val_loss = 0
    total_samples = 0
    class_labels = ['cage', 'inner', 'normal', 'outer', 'roller']
    with torch.no_grad():
        for batch_idx, (_, data, target, _, _) in enumerate(val_loader):
            data = data_augmentation(data)
            data = np.array(data)
            mel_spect = librosa.feature.melspectrogram(y=data, sr=48000, n_fft=2048, hop_length=1024)
            data = librosa.power_to_db(mel_spect, ref=np.max)
            data = torch.from_numpy(data)
            data = torch.unsqueeze(data, 1)
            data, target = data.to(device), target.to(device)
            data = data.float()
            output = model(data)  # [16,5]
            target = target.squeeze()  # target [16,5]

            min_val = torch.min(output)
            max_val = torch.max(output)
            # 对output数据进行归一化
            output = torch.div(torch.sub(output, min_val), (max_val - min_val))
            min_val = torch.min(target)
            max_val = torch.max(target)
            # 对target数据进行归一化
            target = torch.div(torch.sub(target, min_val), (max_val - min_val))

            loss = criterion(output, target)  # 计算损失
            val_loss += loss.item() * data.size(0)  # 累计验证损失
            total_samples += data.size(0)  # 更新总样本数

            max_pre_ind = torch.argmax(output, dim=1)
            max_tar_ind = torch.argmax(target, dim=1)

            # 获取对应的标签值
            pre_labels = [class_labels[index] for index in max_pre_ind.tolist()]
            tar_labels = [class_labels[index] for index in max_tar_ind.tolist()]

            accuracy = accuracy_score(pre_labels, tar_labels)
            precision = precision_score(pre_labels, tar_labels,
                                        average='micro')  # precision_score是一个用于计算分类模型的精确率的函数 micro表示使用微平均计算精确率
            recall = recall_score(pre_labels, tar_labels, average='micro')
            f1 = f1_score(pre_labels, tar_labels, average='micro')
        val_loss /= total_samples
        print(
            f"Val Loss: {val_loss:.4f} | Val Accuracy: {accuracy:.4f}| Recall: {recall:.4f}| Precision: {precision:.4f}| F1 Score: {f1:.4f}")

EPOCHS=5
#writer = SummaryWriter("Training parameters")

# loaded_model = torch.load('F:\CRNN\CRNN\CRNNmodel.pth')#加载模型
for epoch in range(1, EPOCHS + 1):
    train(model, criterion, train_loader,val_loader, optimizer, epoch, device)
# torch.save(model, 'F:\CRNN\CRNN\CRNNmodel.pth')#保存模型

#writer.close()
