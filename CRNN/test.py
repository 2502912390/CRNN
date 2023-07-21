import numpy as np
import scipy.io.wavfile as wav
import glob
import torch
from train import data_augmentation
import torch.nn.functional as F
import librosa
from train import model
from train import device

test_wav = glob.glob('F:\旋转机械故障诊断挑战赛公开数据/测试集/*.wav')
test_wav.sort()

test_wav_feat = []
for path in test_wav:
    sample_rate, audio_data = wav.read(path)
    audio_data = audio_data.reshape(-1)
    test_wav_feat.append(audio_data)

#test_dataset [mat, wav, pad_mask, idx]
pre_labels=[]
class_labels = ['cage','inner', 'normal','outer','roller']
def test(model, test_wav_feat , device):
    print("testing1")
    model.eval()#评估模式
    for data in test_wav_feat:
        data = torch.from_numpy(data).float()
        data = F.normalize(data, dim=-1)#归一化
        data=data_augmentation(data)
        data = np.array(data)
        mel_spect = librosa.feature.melspectrogram(y=data, sr=48000, n_fft=2048, hop_length=1024)
        data = librosa.power_to_db(mel_spect, ref=np.max)
        data=torch.from_numpy(data)
        data = torch.unsqueeze(data, 1)
        data= data.to(device)
        data = data.float()#torch.Size[128, 1, 188]
        data = data.permute(1, 0, 2) # [1, 128, 188]
        data = torch.unsqueeze(data, dim=0)#[1, 1, 128, 188]
        output = model(data)# [1,5]
        # 对output数据进行归一化
        min_val = torch.min(output)
        max_val = torch.max(output)
        output = torch.div(torch.sub(output, min_val), (max_val - min_val))
        max_pre_ind = torch.argmax(output, dim=1)
        pre_label = [class_labels[index] for index in max_pre_ind.tolist()]
        pre_labels.append(pre_label)
test(model, test_wav_feat, device)
