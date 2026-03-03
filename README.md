# 📡 FMCW MIMO Radar Simulation  
## Multi-Target Tracking using MUSIC AOA + IMM (CV/CA)

---

# 1. Project Overview

This project implements a complete end-to-end FMCW MIMO radar perception and multi-target tracking system in Python.

The system simulates a 77 GHz automotive radar and performs:

- FMCW beat signal generation
- Range processing (Fast-Time FFT)
- Doppler processing (Slow-Time FFT)
- 1D CA-CFAR detection
- High-resolution Angle of Arrival estimation (MUSIC)
- Polar-to-Cartesian state transformation
- Covariance propagation using Jacobian
- Global Nearest Neighbor (Hungarian) data association
- Interacting Multiple Model (IMM) filtering
  - Constant Velocity (CV)
  - Constant Acceleration (CA)
- Track birth, confirmation, and deletion logic
- Full animated diagnostic visualization dashboard

The pipeline mirrors a realistic radar perception architecture used in automotive and robotics applications.

---

# 2. Radar System Configuration

## Radar Parameters

- Carrier frequency (fc): 77 GHz
- Bandwidth (B): 200 MHz
- Chirp duration (T): 30 µs
- Sampling frequency (fs): 20 MHz
- Number of chirps per frame: 64
- Number of frames: 120
- Number of RX antennas: 4
- Maximum detection range: 150 m

## Derived Parameters

- Speed of light (c): 3 × 10^8 m/s
- Wavelength (λ) = c / fc
- Antenna spacing (d) = λ / 2
- Chirp slope (S) = B / T
- Frame time = num_chirps × T

---

# 3. Simulated Scenario

Two targets are simulated:

Target 1:
- Initial range: 55 m
- Velocity: −10 m/s
- Azimuth angle: +15°

Target 2:
- Initial range: 50 m
- Velocity: −10 m/s
- Azimuth angle: −15°

Targets move with constant velocity across frames.

---

# 4. Signal Generation

For each frame:

Range evolution:
R(t) = R0 + v (frame_time + kT)

Beat frequency:
f_b = 2 S R / c

Doppler shift:
f_D = 2 v fc / c

The received signal per antenna includes:
- FMCW beat phase
- Doppler phase
- MIMO steering vector for angle modeling

A Hann window is applied along the fast-time dimension before FFT.

---

# 5. Range Processing (Fast-Time FFT)

- FFT applied along fast-time axis
- Only positive frequencies retained
- Frequency bins converted to physical range
- Range limited to max_range

Output:
- Range profile (averaged across chirps)
- Stored for visualization

---

# 6. Doppler Processing (Slow-Time FFT)

- FFT applied across chirps
- FFT shift applied
- Doppler frequency mapped to velocity:
  v = (f_D c) / (2 fc)

Output:
- Range-Doppler magnitude map (in dB scale)

---

# 7. CFAR Detection (1D CA-CFAR)

A 1D CA-CFAR detector is applied along the range dimension.

Parameters:
- Training cells: 12
- Guard cells: 4
- Threshold offset

Detection rule:
Cell is declared detection if:

signal > mean(training_cells) + offset

Note:
This implementation is 1D only (range dimension).

---

# 8. Angle of Arrival Estimation (MUSIC)

For each detected range bin:

1. Extract RX × chirp snapshot matrix
2. Compute spatial covariance:
   R = X Xᴴ / N
3. Apply diagonal loading
4. Perform eigen-decomposition
5. Separate signal and noise subspaces
6. Compute MUSIC pseudo-spectrum:
   P(θ) = 1 / (aᴴ En Enᴴ a)
7. Select peak angle

Angle grid:
-90° to +90° (361 points)

---

# 9. Measurement Model

Each detection produces polar measurements:

(R, V, θ)

Converted into Cartesian state:

x = R cos(θ)
y = R sin(θ)
vx = V cos(θ)
vy = V sin(θ)

Final measurement vector:
z = [x, y, vx, vy]

---

# 10. Covariance Propagation

Measurement uncertainties defined in polar space:

- σ_R = 0.15 + 0.002 R
- σ_V = 0.2
- σ_θ = (2° in radians) / sqrt(SNR)

Polar covariance:
R_polar = diag(σ_R², σ_V², σ_θ²)

Jacobian J computed analytically for transformation.

Cartesian covariance:
R_cart = J R_polar Jᵀ

---

# 11. Multi-Target Tracking

## 11.1 Data Association (GNN)

For each track and measurement:

1. Compute innovation:
   y = z − x_pred

2. Compute Mahalanobis distance:
   d² = yᵀ S⁻¹ y

3. Build cost matrix
4. Apply Hungarian algorithm
5. Apply gating threshold (25)

---

## 11.2 IMM Filter

Two models:

### Constant Velocity (CV)

State:
[x, y, vx, vy]

Transition:
F_cv

Process noise:
Q_cv

---

### Constant Acceleration (CA)

State:
[x, y, vx, vy, ax, ay]

Transition:
F_ca

Process noise:
Q_ca

---

## IMM Steps

1. Mixing
2. Model-specific prediction
3. Model-specific update
4. Likelihood calculation
5. Model probability update
6. State fusion:
   x_fused = μ_cv x_cv + μ_ca x_ca

---

# 12. Track Management

- Tracks initialized as "tentative"
- Confirmed after ≥ 6 hits
- Deleted after 10 consecutive misses
- Age, hits, and misses tracked
- IMM probabilities logged per track

---

# 13. Visualization Dashboard

Generated animated GIF contains:

1. Beat signal (Chirp 0)
2. Range profile
3. Range-Doppler heatmap
4. Cartesian tracking map
5. IMM model probability evolution

Output file:
FMCW_Dashboard IMM AOA.gif

---

# 14. Dependencies

Required packages:

- numpy
- matplotlib
- scipy
- scikit-learn
- tqdm

Install:

pip install numpy matplotlib scipy scikit-learn tqdm

---

# 15. Current Limitations

- No additive thermal noise (noise commented out)
- No clutter modeling
- Only 1D CFAR
- Static gating threshold
- No track score-based confirmation
- Not optimized for real-time performance

---

# 16. Future Extensions

- 2D CFAR implementation
- Chi-square statistical gating
- Clutter and false alarm simulation
- Micro-Doppler feature extraction
- Classification layer (human vs vehicle)
- Real mmWave hardware integration
- Real-time streaming implementation

---

# Author

Raju Kokatanur  
B.Eng. Sustainable Engineering & Resource Management  
