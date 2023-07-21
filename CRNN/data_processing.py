import os.path
import glob, scipy
from torch.utils.data import Dataset
import soundfile as sf
import pandas as  pd
import math
import numpy as np
import torch
from torch.utils.data import DataLoader

def normalize_wav(wav):
    # 对音频数据进行归一化，将其值缩放到 -1 到 1 之间
    return wav / (torch.max(torch.max(wav), -torch.min(wav)) + 1e-10)

def to_mono(wav, rand_ch=False):
    # 将音频转换为单声道
    if wav.ndim > 1:
        if rand_ch:
            # 如果存在多个声道，随机选择一个声道
            ch_idx = np.random.randint(0, wav.shape[-1] - 1)
            wav = wav[:, ch_idx]
        else:
            # 对所有声道取平均值
            wav = np.mean(wav, axis=-1)
    return wav

def pad_wav(wav, pad_to, encoder):
    # 对音频进行填充，使其长度达到 pad_to
    if len(wav) < pad_to:
        pad_from = len(wav)
        # 使用常数进行填充
        wav = np.pad(wav, (0, pad_to - len(wav)), mode="constant")
    else:
        # 如果音频长度超过了 pad_to，则截断为指定长度
        wav = wav[:pad_to]
        pad_from = pad_to
    # 计算填充位置的索引
    pad_idx = np.ceil(encoder._time_to_frame(pad_from / encoder.sr))
    # 创建填充掩码，长度为 encoder.n_frames，填充位置之后的元素为 1，其余为 0
    pad_mask = torch.arange(encoder.n_frames) >= pad_idx
    return wav, pad_mask

def waveform_modification(filepath, pad_to, encoder):  #
    # 读取音频文件
    wav, _ = sf.read(filepath)
    # 转换为单声道
    wav = to_mono(wav)
    # 对音频进行填充，使其长度达到指定的 pad_to
    wav, pad_mask = pad_wav(wav, pad_to, encoder)

    wav = torch.from_numpy(wav).float()
    wav = normalize_wav(wav)
    # 返回处理后的音频数据和填充掩码
    return wav, pad_mask

class Encoder:
    def __init__(self, labels, audio_len, frame_len, frame_hop, net_pooling=1, sr=16000):
        if type(labels) in [np.ndarray, np.array]:
            labels = labels.tolist()
        self.labels = labels
        self.audio_len = audio_len
        self.frame_len = frame_len
        self.frame_hop = frame_hop
        self.sr = sr
        self.net_pooling = net_pooling
        n_samples = self.audio_len * self.sr
        self.n_frames = int(math.ceil(n_samples/2/self.frame_hop)*2 / self.net_pooling)

    def _time_to_frame(self, time):
        sample = time * self.sr
        frame = sample / self.frame_hop
        return np.clip(frame / self.net_pooling, a_min=0, a_max=self.n_frames)

    def _frame_to_time(self, frame):
        time = frame * self.net_pooling * self.frame_hop / self.sr
        return np.clip(time, a_min=0, a_max=self.audio_len)

    def encode_strong_df(self, events_df):
        # from event dict, generate strong label tensor sized as [n_frame, n_class]
        true_labels = np.zeros((self.n_frames, len(self.labels)))
        for _, row in events_df.iterrows():
            if not pd.isna(row['event_label']):
                label_idx = self.labels.index(row["event_label"])
                onset = int(self._time_to_frame(row["onset"]))
                offset = int(np.ceil(self._time_to_frame(row["offset"])))
                true_labels[onset:offset, label_idx] = 1
        return true_labels

    def encode_weak(self, events):
        # from event dict, generate weak label tensor sized as [n_class]
        labels = np.zeros((len(self.labels)))
        if len(events) == 0:
            return labels
        else:
            for event in events:
                labels[self.labels.index(event)] = 1
            return labels

    def decode_strong(self, outputs):
        #from the network output sized [n_frame, n_class], generate the label/onset/offset lists
        pred = []
        for i, label_column in enumerate(outputs.T):  #outputs size = [n_class, frames]
            change_indices = self.find_contiguous_regions(label_column)
            for row in change_indices:
                onset = self._frame_to_time(row[0])
                offset = self._frame_to_time(row[1])
                onset = np.clip(onset, a_min=0, a_max=self.audio_len)
                offset = np.clip(offset, a_min=0, a_max=self.audio_len)
                pred.append([self.labels[i], onset, offset])
        return pred

    def decode_weak(self, outputs):
        result_labels = []
        for i, value in enumerate(outputs):
            if value == 1:
                result_labels.append(self.labels[i])
        return result_labels

    def find_contiguous_regions(self, array):
        #find at which frame the label changes in the array
        change_indices = np.logical_xor(array[1:], array[:-1]).nonzero()[0]
        #shift indices to focus the frame after
        change_indices += 1
        if array[0]:
            #if first element of array is True(1), add 0 in the beggining
            #change_indices = np.append(0, change_indices)
            change_indices = np.r_[0, change_indices]
        if array[-1]:
            #if last element is True, add the length of array
            change_indices = np.r_[change_indices, array.size]
        #reshape the result into two columns
        return change_indices.reshape((-1, 2))

class LabeledDataset(Dataset):  # 继承 torch.utils.data.Dataset 类，创建自己的数据集类。
    def __init__(self, dataset_dir, return_name, encoder):  # F:\旋转机械故障诊断挑战赛公开数据/训练集
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = 48000 * 4  # encoder.wav_len * self.encoder.sr#需要将样本填充到的目标长度  音频长度*采样率
        self.return_name = return_name  # return_name：指示是否返回样本的名称。这个参数可以是一个布尔值，用于控制是否返回样本的名称。

        # 获取mat和wav文件路径列表
        mat_files = glob.glob(dataset_dir + '/*/*/*.mat')  # 'F:\旋转机械故障诊断挑战赛公开数据/训练集/*/*/*.mat'
        wav_files = glob.glob(dataset_dir + '/*/*/*.wav')
        mat_files.sort()
        wav_files.sort()

        # 加载mat文件并重塑数据
        mat_feats = [scipy.io.loadmat(path)['vib_data'].reshape(-1) for path in mat_files]

        # 加载wav文件数据
        wav_feats = [sf.read(path) for path in wav_files]
        # sf.read(path) 会加载指定路径path的 .wav 文件，并返回一个元组，其中包含两个元素：
        # 第一个元素是音频数据的 NumPy 数组，表示采样的音频信号。
        # 第二个元素是音频数据的采样率，即每秒钟对声音信号进行采样的次数。

        # 得到mat和wav 文件的lable
        labels = [os.path.basename(path).split('_')[-4] for path in mat_files]

        # Construct clip dictionary with file name, path, and label
        clips = {}
        for i, mat_path in enumerate(mat_files):  # enumerate(mat_files) 遍历 mat_files 列表，并同时获取索引i和对应的 .mat 文件路径 mat_path
            filename = os.path.basename(mat_path)  # 获取去掉路径部分的纯文件名  train_audio_cage_rpm500_load0_10.wav
            if filename not in clips.keys():  # 避免重复数据
                clips[filename] = {"mat_path": mat_path, "wav_path": wav_files[i], "label": labels[i]}
        # {'mat1': {'mat_path': 'path/to/mat1', 'wav_path': 'path/to/wav1', 'label': 'label1'}}

        # Dictionary for each clip
        self.clips = clips
        self.clip_list = list(clips.keys())  # 获取所有mat文件的名字
        self.mat_feats = mat_feats
        self.wav_feats = wav_feats

    def __len__(self):
        return len(self.clip_list)  # 所有文件的数量

    def __getitem__(self, idx):  # 根据给定的索引 idx 返回相应的样本
        if torch.is_tensor(idx):  # 检查索引 idx 是否为 torch.Tensor 类型，并将其转换为 Python 列表类型
            idx = idx.tolist()
        filename = self.clip_list[idx]  # mat1

        clip = self.clips[filename]  # {'mat_path': 'path/to/mat1', 'wav_path': 'path/to/wav1', 'label': 'label1'}
        mat_path, wav_path = clip["mat_path"], clip["wav_path"]
        # path/to/mat1  path/to/wav1

        # Load mat features
        mat = self.mat_feats[idx]  # mat特征

        # Load wav features  # 处理后的音频数据
        wav, pad_mask = waveform_modification(wav_path, self.pad_to, self.encoder)

        # label全0矩阵
        # label = torch.zeros(self.encoder.n_frames, len(self.encoder.labels))

        label = torch.zeros(1, len(self.encoder.labels))
        # 创建一个形状为 (self.encoder.n_frames, len(self.encoder.labels)) 的全零的张量，并将其赋值给变量 label
        # n_frames 属性可能存储了编码器处理的音频或视频数据的帧数。
        # labels 属性可能存储了编码器可以识别的不同类别或标签。

        label_encoded = self.encoder.encode_weak([clip["label"]])  # label size: [n_class]  对标签进行编码
        label[0, :] = torch.from_numpy(label_encoded).float()  # label size: [n_frames, n_class]
        # 将编码后的标签数据 label_encoded 转换为 PyTorch 张量，并将其赋值给变量 label 的第一行。

        label = label.transpose(0, 1)  # 维度进行转置

        # Return
        out_args = [mat, wav, label, pad_mask, idx]  # mat = out_args[0]
        if self.return_name:
            out_args.extend([filename, mat_path, wav_path])
        return out_args


class UnlabeledDataset(Dataset):
    def __init__(self, dataset_dir, return_name, encoder):
        self.dataset_dir = dataset_dir
        self.encoder = encoder
        self.pad_to = 48000 * 4
        self.return_name = return_name

        # Get list of mat and wav files
        mat_files = glob.glob(dataset_dir + '/*/*/*.mat')
        wav_files = glob.glob(dataset_dir + '/*/*/*.wav')
        mat_files.sort()
        wav_files.sort()

        # Load mat files and reshape the data
        mat_feats = [scipy.io.loadmat(path)['vib_data'].reshape(-1) for path in mat_files]

        # Load wav files
        wav_feats = [sf.read(path) for path in wav_files]

        # Construct clip dictionary with file name, path, and label
        clips = {}
        for i, mat_path in enumerate(mat_files):
            filename = os.path.basename(mat_path)
            if filename not in clips.keys():
                clips[filename] = {"mat_path": mat_path, "wav_path": wav_files[i]}

        # Dictionary for each clip
        self.clips = clips
        self.clip_list = list(clips.keys())
        self.mat_feats = mat_feats
        self.wav_feats = wav_feats

    def __len__(self):
        return len(self.clip_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.clip_list[idx]
        clip = self.clips[filename]
        mat_path, wav_path = clip["mat_path"], clip["wav_path"]

        # Load mat features
        mat = self.mat_feats[idx]

        # Load wav features
        wav, pad_mask = waveform_modification(wav_path, self.pad_to, self.encoder)

        # Return
        out_args = [mat, wav, pad_mask, idx]
        if self.return_name:
            out_args.extend([filename, mat_path, wav_path])
        return out_args

from collections import OrderedDict

def get_labeldict():
    return OrderedDict({"normal": 0,
                        "cage": 1,
                        "inner": 2,
                        "outer": 3,
                        "roller": 4,
                        })
def get_encoder(LabelDict):
    return Encoder(list(LabelDict.keys()),
                   audio_len=78,
                   frame_len=2048,
                   frame_hop=512,
                   net_pooling=1,
                   sr=48000)

dataset_dir="F:\旋转机械故障诊断挑战赛公开数据\训练集"
val_dataset_dir="F:\旋转机械故障诊断挑战赛公开数据\验证集"
test_dataset_dir="F:\旋转机械故障诊断挑战赛公开数据\测试集"

return_name=False

LabelDict=get_labeldict()

encoder = get_encoder(LabelDict)
dataset = LabeledDataset(dataset_dir, return_name, encoder)
val_dataset=LabeledDataset(val_dataset_dir, return_name, encoder)
test_dataset= UnlabeledDataset(test_dataset_dir, return_name, encoder)

batch_size = 16
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)#验证集