# 导入其他需要的库
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# %matplotlib inline
import random
import pickle
import glob
from tqdm import tqdm
import os

# 这个设置是启动Tensorflow的XLA，但不知道是啥（
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
# 关闭TF日志输出，只显示输出error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 导入语音处理相关的库
from python_speech_features import mfcc
import scipy.io.wavfile as wav
import librosa
import librosa.display
from IPython.display import Audio
# 导入所需的模块和类
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Activation, Lambda, Add, Multiply, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import load_model
import pickle

# 使用glob匹配所有以.trn为扩展名的文件路径
# 返回值为列表（绝对路径）
# .trn文件保存对应wav文件的语音文字和拼音
text_paths = glob.glob('G:/data_thchs30/data_thchs30/data/*.trn')

# 获取匹配到的文件总数
total = len(text_paths)

# 打印总文件数
print(total)

# 使用with语句打开第一个匹配到的文件
with open(text_paths[0], 'r', encoding='utf8') as fr:
    # 读取文件中的所有行并存储在lines列表中
    lines = fr.readlines()

    # 打印读取的行
    print(lines)

# 初始化空列表，用于存储处理后的文本和文件路径
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

# 打印第一个文件路径和对应的文本内容
print(paths[0], texts[0])

# mfcc特征维度
mfcc_dim = 13

# librosa库是专用于音频信号处理的第三方库
# 一些介绍
# http://d.glf2ym.cn/xiZxN4
""" librosa.load(path, sr, mono, offset, duration, dtype, res_type)
参数：
    path ：音频文件的路径。
    sr：采样率，是每秒传输的音频样本数，以Hz或kHz为单位。默认采样率为22050Hz（sr缺省或sr=None），高于该采样率的音频文件会被下采样，低于该采样率的文件会被上采样。
    以44.1KHz重新采样：librosa.load(audio_path, sr=44100)
    禁用重新采样（使用音频自身的采样率）：librosa.load(audio_path, sr=None)
    mono ：bool值，表示是否将信号转换为单声道。mono=True为单声道，mono=False为stereo立体声            
    offset ：float，在此时间之后开始阅读（以秒为单位）
    duration：持续时间，float，仅加载这么多的音频（以秒为单位）
    dtype：返回的音频信号值的格式，似乎只有float和float32
    res_type：重采样的格式
返回：
    y：音频时间序列，类型为numpy.ndarray
    sr：音频的采样率，如果参数没有设置返回的是原始采样率
"""


def load_and_trim(path):
    # 加载信号，详细见上
    audio, sr = librosa.load(path)
    # 求信号每帧rms(均方根)，一般信号能量以此衡量, 帧长默认为2048， 因此energy是一组序列
    energy = librosa.feature.rms(y=audio)
    # np.nonzero()返回输入的非零索引
    frames = np.nonzero(energy >= np.max(energy) / 5)
    # 获取能量较大帧的音频信号
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr


# 简单易懂的函数，可视化信号，输入索引，返回信号和其mfcc图
def visualize(index):
    path = paths[index]
    text = texts[index]
    print('Audio Text:', text)

    audio, sr = load_and_trim(path)
    plt.figure(figsize=(12, 3))
    plt.plot(np.arange(len(audio)), audio)
    plt.title('Raw Audio Signal')
    plt.xlabel('Time')
    plt.ylabel('Audio Amplitude')
    plt.show()

    # 获取mfcc特征 短时傅里叶变换长度为551
    feature = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)
    print('Shape of MFCC:', feature.shape)

    # Plot MFCC spectrogram with coordinates
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(feature, sr=sr)

    plt.title('Normalized MFCC')
    plt.ylabel('Time')
    plt.xlabel('MFCC Coefficient')
    plt.colorbar(format='%+2.0f dB')
    # Manually set x-axis tick labels for MFCC coefficients
    num_coefficients = feature.shape[0]
    plt.xticks(np.arange(0, 13), np.arange(1, 13 + 1))

    # Manually set y-axis tick labels for time
    num_frames = feature.shape[0]
    print(num_frames)
    time_in_seconds = librosa.frames_to_time(np.arange(0, num_frames, 100), sr=sr)
    time_labels = [t for t in time_in_seconds]
    plt.yticks(np.arange(0, num_frames, 100))

    plt.tight_layout()
    plt.show()

    return path


# 理论上会播放音频，
# 但是我没听到，可能是pycharm的问题
Audio(visualize(0))


# # 本段代码用于获取features，我提前保存好到features.txt中（应该用pkl后缀规范一点）
# features = []

# 使用tqdm来显示循环进度
# for i in tqdm(range(total)):
#     # 获取当前索引的音频文件路径
#     path = paths[i]
#
#     # 加载和修剪音频
#     audio, sr = load_and_trim(path)
#
#     # 计算音频的MFCC特征并添加到features列表中
#     features.append(mfcc(audio, sr, numcep=mfcc_dim, nfft=551))

# 保存文件
def save_variable(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


# 读取文件
def load_variavle(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


# 写入features
# filename = save_variable(features, 'features.txt')
# 读取features，
# 应该也能用pickle.load()和pickle.dump()写入读取
# 代码后面有使用pickle的例子
features = load_variavle('features.txt')

# 打印MFCC特征的数量和第一个特征的形状
print(len(features), features[0].shape)

# 从特征列表中随机抽取100个样本
samples = random.sample(features, 100)

# 将样本堆叠成矩阵
samples = np.vstack(samples)
##############################################
# 计算抽样样本的MFCC均值和标准差
mfcc_mean = np.mean(samples, axis=0)
mfcc_std = np.std(samples, axis=0)
print(mfcc_mean)
print(mfcc_std)

# 对所有特征进行标准化
# 之前忘了结果一直跑不出来测试
features = [(feature - mfcc_mean) / (mfcc_std + 1e-14) for feature in features]

chars = {}

# 统计所有文本中的字符出现频次
for text in texts:
    for c in text:
        chars[c] = chars.get(c, 0) + 1

# 按字符出现频次排序
chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)

# 仅保留字符列表
chars = [char[0] for char in chars]

# 打印字符数量和前100个字符
print(len(chars), chars[:100])

# 创建字符到ID的映射和ID到字符的映射
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}

# 生成数据音频索引
data_index = np.arange(total)
# 打乱索引
np.random.shuffle(data_index)
# 拆分训练集和测试集
# 数据集中没有分开二者，这里手动分一下
train_size = int(0.9 * total)
test_size = total - train_size
train_index = data_index[:train_size]
test_index = data_index[train_size:]

X_train = [features[i] for i in train_index]
Y_train = [texts[i] for i in train_index]
X_test = [features[i] for i in test_index]
Y_test = [texts[i] for i in test_index]

# 如果还分不清除·楚batch，batch_size，epoch，请看链接
# http://d.glf2ym.cn/wQ4Pf4
# 这里设定batch_size = 8，将占用大概4g显存（如果用gpu的话）
# 但是用cpu好像内存会自动调整，可能
batch_size = 8


# 翻译一下，batch生成器
def batch_generator(x, y, batch_size=batch_size):
    offset = 0
    while True:
        offset += batch_size
        # 第一次循环或超出循环
        # 打乱x和y
        if offset == batch_size or offset >= len(x):
            data_index = np.arange(len(x))
            np.random.shuffle(data_index)
            x = [x[i] for i in data_index]
            y = [y[i] for i in data_index]
            offset = batch_size

        # 数据读取
        X_data = x[offset - batch_size: offset]
        Y_data = y[offset - batch_size: offset]

        # 找最长的数据长度，防止batch塞不下
        X_maxlen = max([X_data[i].shape[0] for i in range(batch_size)])
        Y_maxlen = max([len(Y_data[i]) for i in range(batch_size)])

        # 生成batch
        X_batch = np.zeros([batch_size, X_maxlen, mfcc_dim])
        Y_batch = np.ones([batch_size, Y_maxlen]) * len(char2id)
        X_length = np.zeros([batch_size, 1], dtype='int32')
        Y_length = np.zeros([batch_size, 1], dtype='int32')

        for i in range(batch_size):
            X_length[i, 0] = X_data[i].shape[0]
            X_batch[i, :X_length[i, 0], :] = X_data[i]

            # 这里面y是字符串，所以要转化为数字向量(利用char2id字典)
            Y_length[i, 0] = len(Y_data[i])
            Y_batch[i, :Y_length[i, 0]] = [char2id[c] for c in Y_data[i]]

        inputs = {'X': X_batch, 'Y': Y_batch, 'X_length': X_length, 'Y_length': Y_length}
        outputs = {'ctc': np.zeros([batch_size])}

        # yield返回一个生成器(generator),这意味着如果不调用next()或send()方法，该函数不会真的执行
        yield (inputs, outputs)


epochs = 50
num_blocks = 3
filters = 128

# Input是从tf里import的类，从这里开始，我们将开始网络的构建(WaveNet)
# http://d.glf2ym.cn/cHDBHc
# None意味着这个维度大小是可变的，不确定的
# 这里发现X.shape = [None, None, 13]，这是因为X.shape = [batch_size, shape]
# 而batch_size默认为None
X = Input(shape=(None, mfcc_dim,), dtype='float32', name='X')
Y = Input(shape=(None,), dtype='float32', name='Y')
X_length = Input(shape=(1,), dtype='int32', name='X_length')
Y_length = Input(shape=(1,), dtype='int32', name='Y_length')


# 一维卷积
# 参数filters是输出空间的维度 （即卷积中滤波器的输出数量）
# 参数kernel_size表示卷积核的大小
# 参数strides表示步长，步幅 stride 是一个一维的向量，长度为4。形式是[a,x,y,z]，
# 分别代表[batch滑动步长，水平滑动步长，垂直滑动步长，通道滑动步长]，但这是一维的(
# padding则表示填充方式,casual是因果膨胀卷积(我也不知道是啥，还没看论文)，附论文地址 https://arxiv.org/abs/1609.03499
# dilation_rate: 一个整数，或者单个整数表示的元组或列表，指定用于膨胀卷积的膨胀率。
# 找到篇解释膨胀卷积的博客，还不错 http://d.glf2ym.cn/GPcmA2

def custom_conv1d(inputs, filters, kernel_size, dilation_rate):
    return Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding='causal', activation=None,
                  dilation_rate=dilation_rate)(inputs)


# 将每批前一层的激活量标准化，即进行转换，使平均激活量接近0，激活标准差接近1
# (抄的没看懂，来个人解释一下，我猜就是个归一化)
def batchnorm(inputs):
    return BatchNormalization()(inputs)


# 激活函数嗷，不懂的去看定义，池化是常见操作(Relu)
# https://zhuanlan.zhihu.com/p/48776056
def custom_activation(inputs, activation):
    return Activation(activation)(inputs)

# 这波是创建了hf和hg两个神经元，再相乘，再输入到ha神经元中，具体看链接的WaveNet解释
def res_block(inputs, filters, kernel_size, dilation_rate):
    hf = custom_activation(batchnorm(custom_conv1d(inputs, filters, kernel_size, dilation_rate)), 'tanh')
    hg = custom_activation(batchnorm(custom_conv1d(inputs, filters, kernel_size, dilation_rate)), 'sigmoid')
    h0 = Multiply()([hf, hg])

    # 核大小是1，步长也是1，我只能说6，这不就是序列翻转再乘个值吗
    ha = custom_activation(batchnorm(custom_conv1d(h0, filters, 1, 1)), 'tanh')
    hs = custom_activation(batchnorm(custom_conv1d(h0, filters, 1, 1)), 'tanh')

    return Add()([ha, inputs]), hs


h0 = custom_activation(batchnorm(custom_conv1d(X, filters, 1, 1)), 'tanh')
shortcut = []
for i in range(num_blocks):
    for r in [1, 2, 4, 8, 16]:
        h0, s = res_block(h0, filters, 7, r)
        shortcut.append(s)

h1 = custom_activation(Add()(shortcut), 'relu')
h1 = custom_activation(batchnorm(custom_conv1d(h1, filters, 1, 1)), 'relu')
# 为什么len(char2id) 要加1捏，这是为了防止出现不在字典中的字，我们用空格代替这种情况
Y_pred = custom_activation(batchnorm(custom_conv1d(h1, len(char2id) + 1, 1, 1)), 'softmax')
# 接下来我们最终得到的模型应该是下面这个
sub_model = Model(inputs=X, outputs=Y_pred)


# 计算ctc损失
# 明天再写
def calc_ctc_loss(args):
    y, yp, ypl, yl = args
    return ctc_batch_cost(y, yp, ypl, yl)


# 定义损失计算层
ctc_loss = Lambda(calc_ctc_loss, output_shape=(1,), name='ctc')([Y, Y_pred, X_length, Y_length])
model = Model(inputs=[X, Y, X_length, Y_length], outputs=ctc_loss)
# 定义优化器：随机梯度下降法
optimizer = SGD(learning_rate=0.02, momentum=0.9, nesterov=True, clipnorm=5)

# 这个lambda和上面不同，这个是py自带的一种隐藏函数的定义表明ctc这个函数接受ctc_true和ctc_pred，将返回ctc_pred
model.compile(loss={'ctc': lambda ctc_true, ctc_pred: ctc_pred}, optimizer=optimizer)

filepath = "asr_1.h5"
# model = load_model(filepath, custom_objects=custom_objects, compile=False)

# 3个回调函数，分别是每个epoch保存模型
# 检测loss是否变大，变大直接停止
# 检测var_loss是否变大，变大8次停止
checkpointer = ModelCheckpoint(filepath=filepath, verbose=0)
lr_decay = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=1, min_lr=0.000)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# 使用 Model.fit 方法调整您的模型参数并最小化损失
history = model.fit(
    x=batch_generator(X_train, Y_train),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=batch_generator(X_test, Y_test),
    validation_steps=len(X_test) // batch_size,
    callbacks=[checkpointer, lr_decay, early_stopping])

# 模型可视化
model.summary()
# 存模型
sub_model.save('sub_asr.h5')

plot_model(sub_model, show_shapes=True, dpi=180)

sub_model.save('sub_asr_1.h5')

train_loss = history.history['loss']
valid_loss = history.history['val_loss']
plt.plot(np.linspace(1, len(train_loss), len(train_loss)), train_loss, label='train')
plt.plot(np.linspace(1, len(valid_loss), len(valid_loss)), valid_loss, label='valid')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

with open('dictionary.pkl', 'wb') as fw:
    pickle.dump([char2id, id2char, mfcc_mean, mfcc_std], fw)

from tensorflow.keras.models import load_model
import pickle

with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)

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
