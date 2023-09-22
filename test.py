import os
import pickle
import numpy as np
import glob

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


text_paths = glob.glob('G:/data_thchs30/data_thchs30/data/*.trn')


# 获取匹配到的文件总数
total = len(text_paths)

texts = []
paths = []

# 遍历匹配到的文件路径
for path in text_paths:
    # 使用with语句打开文件
    with open(path, 'r', encoding='utf8') as fr:
        # 读取文件中的所有行并存储在lines列表中
        lines = fr.readlines()

        # 提取第一行文本并进行处理，去除换行符和空格
        line = lines[0].strip('\n').replace(' ', '')

        # 将处理后的文本添加到texts列表中
        texts.append(line)

        # 将处理后的文件路径添加到paths列表中，去除文件扩展名
        paths.append(path.rstrip('.trn'))


def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


features = load_variavle('features.txt')

# 从特征列表中随机抽取100个样本
with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)

features = [(feature - mfcc_mean) / (mfcc_std + 1e-14) for feature in features]
chars = {}

# 统计所有文本中的字符出现频次
for text in texts:
    for c in text:
        chars[c] = chars.get(c, 0) + 1

data_index = np.arange(total)
np.random.shuffle(data_index)
train_size = int(0.9 * total)
test_size = total - train_size
train_index = data_index[:train_size]
test_index = data_index[train_size:]

X_train = [features[i] for i in train_index]
Y_train = [texts[i] for i in train_index]
X_test = [features[i] for i in test_index]
Y_test = [texts[i] for i in test_index]

sub_model = load_model('sub_asr_1.h5')


def random_predict(x, y):
    index = np.random.randint(len(x))
    feature = x[index]
    text = y[index]

    pred = sub_model.predict(np.expand_dims(feature, axis=0))
    pred_ids = K.eval(K.ctc_decode(pred, [feature.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
    pred_ids = pred_ids.flatten().tolist()

    print('True transcription:\n-- ', text, '\n')
    # 防止音频中出现字典中不存在的字，返回空格代替
    print('Predicted transcription:\n--  ' + ''.join([id2char.get(i, ' ') for i in pred_ids]), '\n')


random_predict(X_train, Y_train)
random_predict(X_test, Y_test)
