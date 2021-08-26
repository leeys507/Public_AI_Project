import requests

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

def speech_to_text(file_path, save_txt=False):
    with open(file_path, 'rb') as f:
        stream_file = f.read()

    response = requests.post(endpoint, headers=headers, params=params, data=stream_file)
    if response.status_code != 200: return None

    result = response.json()
    print(result['DisplayText'])
    if save_txt:
        for_save_name = file_path.split('/')[-1]

        with open(save_path + for_save_name + '.txt', 'w') as wf: # UnicodeEncodeError
            wf.write(result['DisplayText'])
    return result['DisplayText']

speech_to_text(test_filename, True)