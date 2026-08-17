# Real-Time Pipelined FIR Voice Processing

A real-time digital signal processing project implemented in **Python** using **FIR filters** and **PyAudio**. The project combines FIR filter design, block-based convolution, envelope detection and signal modulation to transform a microphone input into a metallic/robotic voice effect.

---

## Overview

This project is the continuation of a study on the **design and synthesis of FIR filters using the window method**.

The work is divided into two complementary parts:

1. **FIR filter design and analysis**  
   A Jupyter Notebook develops the FIR design methodology from the ideal frequency response to a causal finite-length filter. Different cutoff frequencies, filter orders and window functions are studied.

2. **Real-time DSP pipeline**  
   The designed FIR filters are integrated into a Python/PyAudio application that processes microphone audio block by block and produces a metallic voice effect in real time.

---

## 1. FIR Filter Design

### 1.1 Why FIR filters?

A Finite Impulse Response (FIR) filter is defined by a finite set of coefficients:

$$
y(n)=\sum_{k=0}^{N-1}h(k)x(n-k)
$$

With symmetric coefficients, FIR filters can provide **linear phase**, resulting in a constant group delay across the frequency components.

The project uses the **window method** to obtain a realizable FIR filter from an ideal frequency-domain specification.

### 1.2 Window method

The design follows these steps:

```text
Ideal frequency response
          ↓
Inverse Fourier Transform
          ↓
Infinite sinc impulse response
          ↓
Windowing / truncation
          ↓
Time shift
          ↓
Causal FIR filter
```

The ideal low-pass impulse response is based on:

$$
g(k)=2f_{cr}\,\mathrm{sinc}(2f_{cr}k)
$$

where:

- $f_{cr}=f_c/f_e$ is the normalized cutoff frequency
- $f_c$ is the cutoff frequency
- $f_e$ is the sampling frequency

The infinite impulse response is truncated using a finite window and shifted to obtain a causal filter.

### 1.3 Window comparison

Two windows are considered:

- **Rectangular window** — narrower transition band for a given order, but stronger ripples.
- **Hamming window** — reduces the sidelobes and Gibbs oscillations at the cost of a slightly wider transition band.

Increasing the filter order $N$ improves selectivity, but also increases computational cost and group delay.

For the design study, **N = 41 with a Hamming window** provided a good compromise between selectivity and ripple level.

---

## 2. Filter Design Examples

The notebook validates the design using a test signal sampled at **20 kHz**:

$$
x(n)=4\sin(2\pi 3000nT_e)+10\sin(2\pi 7500nT_e)
$$

Two filters are designed and evaluated.

### Low-pass FIR

A low-pass filter with:

- Sampling frequency: **20 kHz**
- Cutoff frequency: **4 kHz**
- Order: **N = 41**
- Window: **Hamming**

The filter preserves the 3 kHz component while attenuating the 7.5 kHz component.

### Band-pass FIR

A band-pass filter is constructed as the difference between two low-pass filters:

$$
h_{BP}(n)=h_{high}(n)-h_{low}(n)
$$

For the validation:

- Pass-band: **6–9 kHz**
- Target component: **7.5 kHz**
- Order: **N = 41**
- Window: **Hamming**

The resulting filter isolates the 7.5 kHz component while strongly attenuating the 3 kHz component.

---

## 3. Real-Time Voice Processing Pipeline

The second part applies FIR filtering to a real microphone signal.

The processing chain is:

```text
Microphone
    │
    ▼
Input Band-Pass FIR
    │
    ▼
Envelope Detector
    │
    ├──────────────┐
    │              │
    ▼              ▼
Voice Envelope   Square Wave
    │              │
    └──────┬───────┘
           ▼
       Modulation
           │
           ▼
Output Band-Pass FIR
           │
           ▼
      Re-mixing
           │
           ▼
        Output
```

### Processing stages

#### 1. Input band-pass filtering

The microphone signal is first filtered using a band-pass FIR filter configured for:

**100–300 Hz**

This extracts a selected frequency band from the voice signal.

#### 2. Envelope detection

The filtered signal is processed in two steps:

1. Half-wave rectification
2. Low-pass FIR smoothing

The envelope filter uses:

- Cutoff frequency: **50 Hz**
- FIR length: **21 coefficients**
- Hamming window

This extracts the slow amplitude variations of the vocal signal.

#### 3. Square-wave generation

A square wave is generated with:

- Fundamental frequency: **500 Hz**
- Amplitude: **2.0**

The square wave contains odd harmonics, which are used to modify the spectral content of the voice.

The absolute sample index is maintained between blocks to avoid phase discontinuities.

#### 4. Modulation

The detected vocal envelope is multiplied by the square wave:

$$
s_{mod}(n)=e(n)\cdot s_{square}(n)
$$

This modulation introduces harmonic components responsible for the characteristic metallic/robotic sound.

#### 5. Output band-pass filtering

The modulated signal is filtered with a second FIR band-pass filter:

**100–800 Hz**

This controls the resulting spectral content.

#### 6. Re-mixing

Finally, the processed signal is mixed with the original input:

$$
y(n)=x(n)+y_{BP}(n)
$$

The resulting signal is sent back to the audio output.

---

## 4. Real-Time Audio Architecture

The implementation uses **PyAudio / PortAudio** with frame-based processing.

Audio is processed in blocks of:

| Parameter | Value |
|---|---:|
| Sampling rate | 44.1 kHz |
| Channels | 2 (stereo) |
| Block size | 256 samples |
| Block duration | ≈ 5.8 ms |
| FIR window | Hamming |
| FIR length | 21 coefficients |

A block of 256 samples corresponds to approximately:

$$
\frac{256}{44100}\approx5.8\text{ ms}
$$

This provides a practical compromise between processing load and interactive latency.

---

## 5. Block-Based FIR Convolution

Because the audio stream is processed block by block, the convolution must preserve samples from the previous block.

An **overlap buffer** containing the previous $N-1$ samples is maintained for each FIR filter.

```text
Previous block tail
       │
       ▼
┌───────────────┐
│ Overlap Buffer│
└───────┬───────┘
        │
        +──────── Current block
                  │
                  ▼
             FIR convolution
                  │
                  ▼
             Filtered block
```

Three overlap buffers are used:

- `overlap_bp_in`
- `overlap_env`
- `overlap_bp_out`

Without these buffers, discontinuities can appear at block boundaries, producing audible clicks or gaps.

---

## 6. Software Structure

The main processing functions are organized as follows:

| Function | Purpose |
|---|---|
| `sinc_coeff()` | Computes ideal sinc coefficients |
| `make_fir()` | Builds a causal FIR filter |
| `make_bandpass_fir()` | Builds a band-pass FIR filter |
| `apply_fir()` | Performs block-based FIR convolution |
| `envelope_detector()` | Extracts and smooths the signal envelope |
| `square_wave_generator()` | Generates the modulation carrier |
| `callback_process()` | Executes the complete real-time DSP pipeline |

The `callback_process()` function performs the five main operations:

```text
Input
  ↓
Band-pass FIR
  ↓
Envelope detection
  ↓
Envelope × square wave
  ↓
Output band-pass FIR
  ↓
Mix with original
  ↓
Output
```

---

## 7. Validation

Before enabling the complete DSP chain, the audio path was tested in **pass-through mode** to validate:

- PyAudio stream configuration
- 44.1 kHz sampling
- Stereo channel handling
- `bytes → NumPy float32` conversion
- Microphone selection
- Block-based processing

The complete pipeline was then tested with a live microphone signal.

The resulting voice exhibits a **metallic and robotic timbre**, produced by the interaction between the extracted vocal envelope and the harmonic content of the square-wave modulation.

With the test configuration:

- Input band: **100–300 Hz**
- Carrier frequency: **500 Hz**
- Output band: **100–800 Hz**
- Envelope cutoff: **50 Hz**

the metallic effect is clearly perceptible.

---

## 8. Repository Structure

A recommended repository structure is:

```text
.
├── README.md
├── notebooks/
│   └── FIR_Filter_Design.ipynb
├── src/
│   └── voice_pipeline.py
├── assets/
│   └── pipeline.png
└── docs/
    └── reports/
        ├── rapport_TNS_n1.pdf
        └── rapport_TNS_n2.pdf
```

The **Jupyter Notebook** focuses on the mathematical design and analysis of the FIR filters, while the Python source code contains the real-time processing pipeline.

---

## 9. Technologies

- **Python**
- **NumPy**
- **Matplotlib**
- **PyAudio**
- **PortAudio**
- Digital Signal Processing
- FIR filter design
- Window method
- Real-time audio processing
- Block-based convolution

---

## 10. Key Concepts Demonstrated

This project combines theoretical DSP concepts with a practical real-time application:

- FIR filter synthesis
- Linear-phase FIR filters
- Windowing method
- Sinc impulse response
- Hamming and rectangular windows
- Filter order / selectivity trade-off
- Band-pass filter synthesis
- Discrete convolution
- Overlap buffering
- Envelope detection
- Signal modulation
- Frame-based real-time DSP
- Audio streaming with PyAudio

---

## Documentation

The detailed mathematical development and experimental results are available in the project reports.

- **TP 1:** FIR filter synthesis using the window method
- **TP 2:** Real-time FIR-based voice processing pipeline

The Jupyter Notebook contains the practical FIR design study and can be used to reproduce the filter coefficient calculations and frequency-response analysis.

---

## Author

**Rayen BOUAFIF**  
Engineering Student — Embedded Systems & Telecommunications  
INSA Hauts-de-France / INSAT

2025–2026
