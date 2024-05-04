import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import librosa
from matplotlib.lines import Line2D
from tqdm import tqdm
from tensorflow.keras.models import load_model
import base64
from io import BytesIO

n_mels = 128
sr = 22050
hop_length = 512
window_size = 2048
label_map = {
    
    0: "Asthma",
    1: "Bronchiectasis",
    2: "Bronchiolitis",
    3: "Bronchitis",
    4: "COPD",
    5: "Lung Fibrosis",
    6: "Pleural Effusion",
    7: "Pneumonia",
    8: "URTI",
    9: "Healthy",
}
def plot_maps(img1, img2,vmin=0.3,vmax=0.7, mix_val=2):
    f = plt.figure(figsize=(45,135))
    plt.subplot(1,3,1)
    plt.imshow(img1,vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.title("Saliency Map")
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.axis("off")
    plt.title("Original Image")
    plt.subplot(1,3,3)
    plt.imshow(img1*mix_val+img2/mix_val )
    plt.axis("off")
    plt.title("Highlights")
    
def norm_flat_image(img):
    grads_norm = img[:,:,0]+ img[:,:,1]+ img[:,:,2]
    grads_norm = (grads_norm - tf.reduce_min(grads_norm))/ (tf.reduce_max(grads_norm)- tf.reduce_min(grads_norm))
    return grads_norm
    
def getExplainablePred(spectrogram, audio, model_type): 
    
    # Convert the input image to a TensorFlow tensor
    input_img = tf.convert_to_tensor(spectrogram, dtype=tf.float32)
    input_img = tf.expand_dims(input_img, axis=0)
    loaded_model = load_model('./models/disease_model_v1.h5') if model_type == 2 else load_model('./models/binary_model_v2.h5')
    # watch the weights
    with tf.GradientTape() as tape:
        tape.watch(input_img)
        result = loaded_model(input_img)
        max_score = result[0, tf.argmax(result, axis=1)[0]]

    grads = tape.gradient(max_score, input_img)
    # plot_maps(norm_flat_image(grads[0]), norm_flat_image(input_img[0]))
    
    prediction = int(tf.argmax(result, axis=1)[0])
    
    norm_flat_grad = norm_flat_image(grads[0])

    low_th = 0.7
    avg_th = 0.8
    high_th = 0.9
    
    # get significant regions
    average_sig = tf.logical_and(norm_flat_grad >= avg_th, norm_flat_grad < high_th)
    high_sig = (norm_flat_grad >= high_th)
    low_sig = tf.logical_and(norm_flat_grad >= low_th, norm_flat_grad < avg_th)

    significant_regions_arr = [low_sig, average_sig, high_sig]

    _audio = audio.copy()
    time_vector = np.arange(0, len(_audio)) / sr
    
    # Create a figure and plot the original audio waveform
    plt.figure(figsize=(12, 6))
    plt.plot(time_vector, _audio, label='Original Audio', color='#2559f5')
    
    opasity_arr = [0.9, .9, .9]
    color_arr = ["#f2b72c", "#de2626", "#4d000d"]
    
    
    for significant_regions, opasity, color in zip(significant_regions_arr, opasity_arr, color_arr):
    
        significant_time_indices, significant_frequency_indices = np.where(significant_regions)

        # Map time indices to time in seconds (assuming you have the sampling rate)
        time_in_seconds = significant_time_indices * hop_length / sr


        # getting the highlighted section of the audio
        significant_audio_segments = []
        for t in tqdm(np.unique(time_in_seconds), desc="Itterating time in seconds"):
            start_sample = int(t * sr)
            end_sample = start_sample + window_size  
            
            significant_audio_segments.append(audio[start_sample:end_sample])
            
        
        legend_elements = [Line2D([0], [0], color='#2559f5', label='Original Audio'),
                            Line2D([0], [0], color="#f2b72c", label='Low relavence'),
                            Line2D([0], [0], color="#de2626", label='Average relavence'),
                            Line2D([0], [0], color="#4d000d", label='high relavence')
                            ]

        prev_start_sample = None
        prev_end_sample = 0
        continue_segment = []
        
        # Highlight and plot the significant audio segments
        for t, segment in tqdm(zip(np.unique(time_in_seconds), significant_audio_segments), desc="Plotting"):
            start_sample = int(t * sr)
            end_sample = start_sample + len(segment)
            
            if prev_start_sample == None:
                prev_start_sample = start_sample
                prev_end_sample = end_sample
            else:
                if prev_end_sample >= start_sample: 
                    start_sample = prev_end_sample
                if prev_end_sample < end_sample:
                    prev_end_sample = end_sample
                else:
                    prev_start_sample = None
                    prev_end_sample = 0

            
            if start_sample < end_sample:
                plt.plot(time_vector[start_sample:end_sample], segment[-(end_sample - start_sample):], color=color, alpha=opasity)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.axis('off')
    plt.legend().set_visible(False)
    plt.gca().set_facecolor('none')
    
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', transparent=True)
    img_buffer.seek(0)
    base64Img = base64.b64encode(img_buffer.getvalue()).decode()
    base64Img += '=' * ((4 - len(base64Img) % 4) % 4)
    plt.clf()
    plt.close()
    return base64Img


def generate_mel_spec(audio):
  mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=22050, n_mels=128, n_fft=2048, hop_length=512))
  return mel

def generate_mfcc(audio):
  mfcc = librosa.feature.mfcc(y=audio, n_mfcc=128, n_fft=2048, hop_length=512)
  return mfcc

def generate_chroma(audio):
  chroma = librosa.feature.chroma_stft(y=audio, sr=22050, n_chroma=128, n_fft=2048, hop_length=512)
  return chroma
    
# audio_file_path = './New Audio Binary/'+ "2.wav"
# # audio, sr = librosa.load(audio_file_path, sr=22500, duration=6)    
# y, sr = librosa.load(audio_file_path, sr=22500, duration=6 )
# mean = np.mean(y)
# std = np.std(y)
# norm_audio = (y - mean) / std

# mel = generate_mel_spec(norm_audio)
# mfcc_1 = generate_mfcc(norm_audio)
# chroma_1 = generate_chroma(norm_audio)

# three_chanel = np.stack((mel, mfcc_1, chroma_1), axis=2)

# base64Img=getExplainablePred(three_chanel, norm_audio, 2)

