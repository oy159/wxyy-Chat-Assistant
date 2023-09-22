import pyaudio
import wave
import librosa
import numpy as np
from python_speech_features import mfcc
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


def record_audio(output_wav_path, duration=5, sample_rate=44100, chunk_size=1024):
    audio = pyaudio.PyAudio()

    # 打开麦克风
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording...")

    frames = []

    # 监听并录制音频
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Recording finished.")

    # 关闭麦克风
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # 保存录制的音频为 WAV 文件
    with wave.open(output_wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))


output_wav_path = "./audio/recorded_audio.wav"

if not os.path.exists(output_wav_path):
    record_audio(output_wav_path, duration=5)

with open('dictionary.pkl', 'rb') as fr:
    [char2id, id2char, mfcc_mean, mfcc_std] = pickle.load(fr)


def load_and_trim(path):
    audio, sr = librosa.load(path)
    energy = librosa.feature.rms(y=audio)
    frames = np.nonzero(energy >= np.max(energy) / 5)
    indices = librosa.core.frames_to_samples(frames)[1]
    audio = audio[indices[0]:indices[-1]] if indices.size else audio[0:0]

    return audio, sr


mfcc_dim = 13
sub_model = load_model('sub_asr_1.h5')


def single_predict(audio_path):
    # 加载和修剪音频
    audio, sr = load_and_trim(audio_path)

    # 计算音频的MFCC特征
    feature = mfcc(audio, sr, numcep=mfcc_dim, nfft=551)

    feature = (feature - mfcc_mean) / (mfcc_std + 1e-14)

    pred = sub_model.predict(np.expand_dims(feature, axis=0))
    pred_ids = K.eval(K.ctc_decode(pred, [feature.shape[0]], greedy=False, beam_width=10, top_paths=1)[0][0])
    pred_ids = pred_ids.flatten().tolist()

    print('Predicted transcription:\n--  ' + ''.join([id2char.get(i, ' ') for i in pred_ids]), '\n')


# Specify the path to your MP3 audio file
audio_path = "./audio/recorded_audio.wav"

single_predict(audio_path)
