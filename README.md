# Pitch Shifting

An exploration of several different audio pitch shifting methods.

## Methods

### OLA (Overlap Add)

As the most basic pitch shifter, OLA creates several overlapping windows, runs them through a window function, and synthesizes a new signal with a larger/smaller window spacing.

### SOLA (Synchronized Overlap Add)

Similar to OLA, SOLA creates several overlapping windows and uses them to synthesize a new signal. The only difference is in choosing the new window position, where it tries to find the lowest autocorrelation (lowest phase difference) in order to minimize discontinuities in the signal.

### Phase Vocoder

The phase vocoder computes the short-time fourier transform (STFT) of the signal, interpolating the magnitude and phase differences along the time axis. After, it reconstructs the phases by summing the phase differences and applies the inverse STFT to the synthesized STFT. To retain percussive sounds, the phase vocoder also resets the phase summation when high transience is detected.

## Acknowledgements

Code for the phase vocoder is based on [JentGent's pitch shifting walkthrough](https://github.com/JentGent/pitch-shift)
