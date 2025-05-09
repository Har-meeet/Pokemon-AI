import cv2
import numpy as np
import matplotlib.pyplot as plt

def Dialogue_Classification(img):
    # Load and convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Fourier Transform
    spectrum = fourier_transform(img)

    # Classify the screen
    classification = classify_screen(spectrum)

    return classification

def fourier_transform(img):
    # Ensure float32 type
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
    return magnitude_spectrum

def analyze_high_frequencies(spectrum, cutoff_ratio=0.2):
    """
    Analyzes the high-frequency content in the outer regions of the spectrum.
    cutoff_ratio: The proportion of pixels from the edge to consider as high-frequency.
    """
    h, w = spectrum.shape
    h_cutoff = int(h * cutoff_ratio)
    w_cutoff = int(w * cutoff_ratio)

    # Define the four high-frequency corners
    top_left = spectrum[:h_cutoff, :w_cutoff]
    top_right = spectrum[:h_cutoff, -w_cutoff:]
    bottom_left = spectrum[-h_cutoff:, :w_cutoff]
    bottom_right = spectrum[-h_cutoff:, -w_cutoff:]

    # Combine all high-frequency regions
    high_freq_region = np.concatenate([top_left, top_right, bottom_left, bottom_right], axis=None)

    # Calculate the high-frequency intensity ratio
    high_freq_ratio = np.sum(high_freq_region > 100) / high_freq_region.size

    # Calculate the mean intensity of the high-frequency regions
    high_freq_mean = np.mean(high_freq_region)

    return high_freq_ratio, high_freq_mean

def classify_screen(spectrum):
    # Analyze high-frequency content
    high_freq_ratio, high_freq_mean = analyze_high_frequencies(spectrum, cutoff_ratio=0.25)

    # Classification logic:
    # Dialogue screens are expected to have high high-frequency ratios and mean intensity
    if high_freq_ratio > 0.996 and high_freq_mean > 152:
        print (f"High Frequency Ratio: {high_freq_ratio:.4f}, Mean Intensity: {high_freq_mean:.2f}")
        return True
    else:
        return False
