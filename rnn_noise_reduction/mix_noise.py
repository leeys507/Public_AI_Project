'''
need to specify clean_file, noise_file, output_noisy_file path
'''

import argparse
import array
import math
import numpy as np
import random
import wave
import os
import glob

clean_file_paths = glob.glob(os.path.join('./clean_file', '*'))
noise_file_paths = glob.glob(os.path.join('./noise_file', '*', '*.wav'))
output_noisy_path = './output_noisy_file/'

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--snr', type=float, default='0', required=True) # 숫자가 커질수록 선명한 소리가 남음
#     args = parser.parse_args()
#     return args

def cal_adjusted_rms(clean_rms, snr):
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

def cal_amp(wf):
    buffer = wf.readframes(wf.getnframes())
    amptitude = (np.frombuffer(buffer, dtype="int16")).astype(np.float64)
    return amptitude

def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

if __name__ == '__main__':
    # args = get_args()
    for clean_file in clean_file_paths:
        snr = random.randrange(-10, 11)
        noise_file = noise_file_paths[random.randrange(len(noise_file_paths))]
        clean_wav = wave.open(clean_file, "r")
        noise_wav = wave.open(noise_file, "r")

        clean_amp = cal_amp(clean_wav)
        noise_amp = cal_amp(noise_wav)

        start = random.randint(0, len(noise_amp)-len(clean_amp))
        clean_rms = cal_rms(clean_amp)
        split_noise_amp = noise_amp[start: start + len(clean_amp)]
        noise_rms = cal_rms(split_noise_amp)
        adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
        adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms)
        mixed_amp = (clean_amp + adjusted_noise_amp)

        if (mixed_amp.max(axis=0) > 32767):
            mixed_amp = mixed_amp * (32767/mixed_amp.max(axis=0))
            clean_amp = clean_amp * (32767/mixed_amp.max(axis=0))
            adjusted_noise_amp = adjusted_noise_amp * (32767/mixed_amp.max(axis=0))

        output_filename = clean_file.split('/')[-1].split('.')[0] \
                          + f"_{noise_file.split('/')[-2]}_" \
                          + noise_file.split('/')[-1].split('.')[0] + f'_{int(snr)}db' + '.wav'

        noisy_wave = wave.Wave_write(output_noisy_path + output_filename)
        noisy_wave.setparams(clean_wav.getparams())
        noisy_wave.writeframes(array.array('h', mixed_amp.astype(np.int16)).tostring())
        noisy_wave.close()