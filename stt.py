import requests
import noisereduce as nr
import scipy.io.wavfile as wavfile
import os
import glob

# key, endpoint
subscription_key = '539b71356bc54e8a9655d3c59e47e625'
endpoint = 'https://koreacentral.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1'

# filename
test_filename = 'data/wave_data/KsponSpeech_000011.wav'
save_path = 'data/label/'

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    # 'Content-type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
}

params = {
    'language': 'ko-KR',
}

def speech_to_text(file_paths, apply_noisereduce=False):
    stt_result = []
    for file_path in file_paths:
        with open(file_path, 'rb') as f:
            stream_file = f.read()

        if apply_noisereduce:
            stream_file = noise_reduction(file_path)

        response = requests.post(endpoint, headers=headers, params=params, data=stream_file)

        if response.status_code != 200:
            print(f"path: {file_path} --> Unknown Speech")
            stt_result.append([file_path, 'Unknown Speech'])
        else:
            result = response.json()
            print(f"path: {file_path} --> {result['DisplayText']}")
            stt_result.append([file_path, result['DisplayText']])
    return stt_result

def folder_to_filepaths(folder_path):
    return glob.glob(os.path.join(folder_path, "*.wav"))


def noise_reduction(file_path):
    rate, data = wavfile.read(file_path)
    # noise_reduced_data = nr.reduce_noise(y=data, sr=rate, n_std_thresh_stationary=1, stationary=True)
    noise_reduced_data = nr.reduce_noise(y=data, sr=rate, thresh_n_mult_nonstationary=1.1)
    wavfile.write('tmp.wav', data=noise_reduced_data, rate=rate)
    with open('tmp.wav', 'rb') as f:
        stream_file = f.read()
    os.remove('tmp.wav')
    return stream_file