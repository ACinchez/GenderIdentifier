import numpy as np
import librosa
from python_speech_features import mfcc, delta
from sklearn import preprocessing


class FeaturesExtractor:
    def extract_features(self, audio_path):
        signal, sample_rate = librosa.load(audio_path, sr=None)
        mfcc_features = mfcc(signal, samplerate=sample_rate, winlen=0.025, winstep=0.01, numcep=13, nfilt=26, nfft=512)
        mfcc_features = preprocessing.scale(mfcc_features)
        deltas = delta(mfcc_features, 2)
        double_deltas = delta(deltas, 2)
        combined = np.hstack((mfcc_features, deltas, double_deltas))
        return combined
