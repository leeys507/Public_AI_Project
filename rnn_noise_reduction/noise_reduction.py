import noisereduce as nr
import scipy.io.wavfile as wavfile

rate, data = wavfile.read('../../datasets/output_noisy_file/1_0092_TMETRO_ch02_-6db.wav') # test file
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write('reducednoise.wav', rate=rate, data=reduced_noise)