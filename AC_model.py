

### Acoustic characterisation of marine energy devices


import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

def calculate_psd(audio_file, window_size=1024, overlap=512):
    # Read audio file
    sampling_rate, audio_data = wav.read(audio_file)
    
    # Calculate PSD
    num_samples = len(audio_data)
    num_windows = int(np.ceil((num_samples - overlap) / (window_size - overlap)))
    
    psd_values = []
    
    for i in range(num_windows):
        start = i * (window_size - overlap)
        end = min(start + window_size, num_samples)
        windowed_data = audio_data[start:end] * np.hamming(end - start)
        
        # Compute the one-dimensional discrete Fourier Transform
        fft_result = np.fft.fft(windowed_data)
        
        # Calculate the Power Spectral Density
        psd = np.abs(fft_result) ** 2 / window_size
        psd_values.append(psd)
    
    # Average the PSD values across all windows
    avg_psd = np.mean(psd_values, axis=0)
    
    # Frequency axis
    freq_axis = np.fft.fftfreq(window_size, 1 / sampling_rate)[:window_size // 2]
    
    return freq_axis, avg_psd

# Example usage
audio_file_path = 'path/to/your/audio/file.wav'
freq_axis, psd_values = calculate_psd(audio_file_path)

# Plot the PSD
plt.plot(freq_axis, 10 * np.log10(psd_values))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.title('Power Spectral Density')
plt.show()
