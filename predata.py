import glob
import librosa
import numpy as np


def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rms(y=audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr


class PreData(object):
    text_paths = glob.glob('G:/data_thchs30/data_thchs30/data/*.trn')
    total = len(text_paths)
    texts = []
    paths = []
    mfcc_dim = 13

    def init_texts_paths(self):
        for path in self.text_paths:
            # 使用with语句打开文件
            with open(path, 'r', encoding='utf8') as fr:
                # 读取文件中的所有行并存储在lines列表中
                lines = fr.readlines()

                # 提取第一行文本并进行处理，去除换行符和空格
                line = lines[0].strip('\n').replace(' ', '')

                # 将处理后的文本添加到texts列表中
                self.texts.append(line)

                # 将处理后的文件路径添加到paths列表中，去除文件扩展名
                self.paths.append(path.rstrip('.trn'))
