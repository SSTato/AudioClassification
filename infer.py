import argparse
import functools
import shutil
import numpy as np
import torch
import time
from data_utils.reader import CustomDataset
from data_utils.reader import load_audio
from modules.ecapa_tdnn import EcapaTdnn
from utils.utility import add_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('use_model',        str,    'ecapa_tdnn',             '所使用的模型')
add_arg('audio_path',       str,    'fout/video_9_.wav', '音频路径')
add_arg('num_classes',      int,    17,                        '分类的类别数量')
add_arg('label_list_path',  str,    'dataset/label_list.txt',  '标签列表路径')
add_arg('model_path',       str,    'output/models/30model.pth','模型保存的路径')
add_arg('feature_method',   str,    'melspectrogram',          '音频特征提取方法', choices=['melspectrogram', 'spectrogram'])
args = parser.parse_args()


train_dataset = CustomDataset(data_list_path=None, feature_method=args.feature_method)
# 获取分类标签
with open(args.label_list_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
class_labels = [l.replace('\n', '') for l in lines]
# 获取模型
device = torch.device("cuda")
if args.use_model == 'ecapa_tdnn':
    model = EcapaTdnn(num_classes=args.num_classes, input_size=train_dataset.input_size)
else:
    raise Exception(f'{args.use_model} 模型不存在!')
model.to(device)
model.load_state_dict(torch.load(args.model_path))
model.eval()

aaa = 0.8  # 阈值


def infer():
    data = load_audio(args.audio_path, mode='infer')
    data = data[np.newaxis, :]
    data = torch.tensor(data, dtype=torch.float32, device=device)
    # 执行预测
    output = model(data)
    result = torch.nn.functional.softmax(output, dim=-1)
    result = result.data.cpu().numpy()
    # 显示图片并输出结果最大的label
    lab = np.argsort(result)[0][-1]
    score = result[0][lab]

    '''
    if score >= aaa:
        if int(class_labels[lab]) in [0, 1, 7, 9, 16, 17]:
            out = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))) + '-正常'
            txt = 'C:/Users/Administrator/Desktop/AudioClassification-Pytorch-master/tout/' + out
            with open(f'{txt}.txt', 'w') as f:
                f.write(out)
        elif int(class_labels[lab]) in [3, 4, 5, 6, 10, 11]:
            out = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))) + '-异常'
            wavout = 'C:/Users/Administrator/Desktop/AudioClassification-Pytorch-master/fout/' + out + '.wav'
            shutil.copy(args.audio_path, wavout)

    elif score < aaa:
        out = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))) + '-异常'
        wavout = 'C:/Users/Administrator/Desktop/AudioClassification-Pytorch-master/fout/' + out + '.wav'
        shutil.copy(args.audio_path, wavout)

        # 保存异常txt
        
        txt = 'C:/Users/Administrator/Desktop/AudioClassification-Pytorch-master/out/' + out
        with open(f'{txt}.txt', 'w') as f:
            f.write(out)
        '''

    print(score)
    print(f'音频：{args.audio_path} 的预测结果标签为：{class_labels[lab]}')


if __name__ == '__main__':
    infer()

