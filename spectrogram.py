import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy
import numpy as np
import tensorflow as tf
from flask import jsonify
import io
from xai import getExplainablePred  
import base64
from io import BytesIO
from model_services import do_primary_prediction, do_secondary_prediction


def generate_mel_spec(audio):
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512))
    return mel


def generate_mfcc(audio):
    mfcc = librosa.feature.mfcc(y=audio, n_mfcc=128, n_fft=2048, hop_length=512)
    return mfcc


def generate_chroma(audio):
    chroma = librosa.feature.chroma_stft(y=audio, sr=22050, n_chroma=128, n_fft=2048, hop_length=512)
    return chroma


def generate_result(audio):
    # Load audio file
    y, sr = librosa.load(audio, sr=22500, duration=6 )
    
    mean = np.mean(y)
    std = np.std(y)
    norm_audio = (y - mean) / std

    mel = generate_mel_spec(norm_audio)
    mfcc_1 = generate_mfcc(norm_audio)
    chroma_1 = generate_chroma(norm_audio)

    three_chanel = np.stack((mel, mfcc_1, chroma_1), axis=2)

    expanded_sample = np.array([three_chanel])

    result = do_primary_prediction(expanded_sample)
    if not result['result']:
        base64xai = getExplainablePred(three_chanel,norm_audio,1)
        result_data = {'diseases': [],'probabilities':[float(result['probability'])], 'xai_base64': base64xai}
        return jsonify(result_data)
    else:

        secondary_result = do_secondary_prediction(expanded_sample)
        floated_probabilities = []
        for i in secondary_result['probabilities']:
            floated_probabilities.append(float(i))
        secondary_result['probabilities'] = floated_probabilities
        base64xai = getExplainablePred(three_chanel,norm_audio,2)
    
        # decoded_img = base64.b64decode(base64xai)
        # img_buffer = BytesIO(decoded_img)
        secondary_result["xai_base64"]=base64xai

        return jsonify(secondary_result)