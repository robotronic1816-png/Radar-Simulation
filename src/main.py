import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation 
from scipy.signal import windows, butter, filtfilt
from sklearn.cluster import DBSCAN



# CFAR FUNCTION
 
def os_cfar_2d(rd_db,
            tr=12, gr=6,
            td=8,  gd=4,
            rank = 0.75,
            offset_db=6.0
):
    D, R = rd_db.shape
    detections = np.zeros_like(rd_db, dtype=int)
    rd_lin = 10**(rd_db / 10)

    for d in range(td+gd, D - (td+gd)):
        for r in range(tr+gr, R - (tr+gr)):

            noise = []

            for dd in range(d - td - gd, d + td + gd + 1):
                for rr in range(r - tr - gr, r + tr + gr + 1):
                    if abs(dd - d) <= gd and abs(rr - r) <= gr:
                        continue
                    noise.append(rd_lin[dd, rr])
        if len(noise) < 10:
            continue                
            
        noise = np.sort(noise)
        k = int(rank * len(noise))
        noise_level = noise[k]
        threshold = 10*np.log10(noise_level) + offset_db
    if rd_db[d, r] >  threshold:
                detections[d, r] = 1

    return detections
 



# RADAR PARAMETERS
c = 3e8
fc = 77e9
B = 200e6
T = 30e-6
Sf = B / T
fs = 20e6

num_chirps = 64
num_frames = 120
SNR_dB = 15
max_range = 150
targets = [
    {"Rn": 50.0, "Vl": -20},
    {"Rn": 65.0, "Vl": -10}
]




print("FMCW radar simulation started")

# Kalman Filter Setup

T_frame = num_chirps * T

F = np.array([[1, T_frame],
             [0,1]])
H = np.eye(2)
Q = np.diag([0.05, 0.5])
R_meas = np.diag([0.1, 1.0])

GT_TH = 9.21
Max_miss = 6
# Track class
class Track:
    def __init__(self, z):
        # z = np.asarray(z).reshape(2,) 
         self.x = z.copy()
         self.P = np.diag([25.0, 9.0])
         self.age = 1
         self.hits = 1
         self.misses = 0
tracks =[]
tracks_hist = []         
hist_signal  = []
hist_range = []
hist_rd_map = []
hist_cfar = []
# Signal Setup
t_fast = np.arange(0, T, 1/fs)
N_fast = len(t_fast)

tx = np.exp(1j * np.pi * Sf * t_fast**2)
f_b_max = 2 * max_range * Sf /c
b, a = butter(4, min(0.9 *fs / 2, f_b_max) / (fs / 2))

print("\nInitiating Multi- frame Tracking  ...\n")

# Frame Loop
for frame in range(num_frames): 
    # FMCW Simu
    beat_matrix = np.zeros((num_chirps, N_fast), dtype=complex)
    
    for k in range(num_chirps):
        #rx_total = 0
        beat = np.zeros(N_fast, dtype=complex)
        for tgt in targets:
            R = tgt["Rn"] + tgt["Vl"] * frame * num_chirps * T
            fb = 2 * Sf * R / c
            fD = 2 * tgt["Vl"] * fc / c
            assert np.isscalar(fb), f"tau is not scalar, shape={np.shape(fb)}"
                
            phase = (
                    2 * np.pi * fb * t_fast
                    + 2 * np.pi * fD * k * T
            )
            beat += np.exp(1j * phase)

            #rx_total += rx
        
        beat = filtfilt(b, a, beat)
        beat_matrix[k, :] = beat
        hist_signal.append(np.real(beat_matrix[0, :]))
# RANGE FFT
    beat_matrix *= windows.hann(len(t_fast))[None, :]
    rng_fft = np.fft.fft(beat_matrix, axis=1)[:, :len(t_fast)//2]

    freqs = np.fft.fftfreq(len(t_fast), d=1/fs)[:len(t_fast)//2]
    ranges = (c * freqs) / (2 * Sf)

    valid = ranges <= max_range
    ranges = ranges[valid]
    rng_fft = rng_fft[:, valid]
    rng_profile = np.mean(np.abs(rng_fft), axis=0)
    hist_range.append(20 * np.log10(rng_profile + 1e-6))

   # rng_fft -= np.mean(rng_fft, axis=0)
   # rng_fft *= windows.blackman(num_chirps)[:, None]
    #rng_fft = rng_fft / (np.linalg.norm(rng_fft, axis=0, keepdims=True) + 1e-6) 
# DOPPLER FFT
    rng_fft -= np.mean(rng_fft, axis=0, keepdims=True)
    rng_fft *= windows.blackman(num_chirps)[:, None]


    doppler_fft = np.fft.fftshift(
        np.fft.fft(rng_fft, axis=0),
        axes=0
)

    rd = np.abs(doppler_fft)
    rd_db = 20 * np.log10(rd + 1e-6)
    hist_rd_map.append(rd_db)
# CFAR
    
    det = os_cfar_2d(
        rd_db,
        tr=12, td=8,
        gr=6, gd=4,
        rank=0.75,
        offset_db=7.0             )
    hist_cfar.append(det)
    d_idx, r_idx = np.where(det == 1)
    vel_axis = np.fft.fftshift(np.fft.fftfreq(num_chirps, d=T)) * c / (2 * fc)

    measurements = []

    if len(r_idx) > 0:
       pts_r = ranges[r_idx]
       pts_v = vel_axis[d_idx] 
       pts_p = rd_db[d_idx, r_idx]
       X = np.column_stack((pts_r, pts_v))
       X_scaled = X.copy()
       X_scaled[:, 0] /= 5.0
       cluster = DBSCAN(eps=1.5, min_samples=3).fit(X_scaled)
       labels = cluster.labels_
# Centroid Extraction
       unq_label =set(labels)
       for lab in unq_label:
           if lab == -1: continue
           mask = (labels == lab)
           w = pts_p[mask]
           w = w - w.min() + 1e-3
           z_r = np.sum(pts_r[mask] * w) / np.sum(w)       
           z_v = np.sum(pts_v[mask] * w) / np.sum(w)
           if abs(z_v) < 1.0:
               continue
           measurements.append(np.array([z_r,z_v]))       

# Multi traget tracking (NN + gating)
        
    for trk in tracks:
        trk.x = F @ trk.x
        trk.P = F @ trk.P @ F.T + Q
        trk.misses += 1
        trk.age += 1
    used = set()
    for trk in tracks:
        S_k = H @ trk.P @ H.T + R_meas
        S_inv = np.linalg.inv(S_k)
        best_d, best_j = np.inf, None

        for j, z in enumerate(measurements):
            if j in used:
                continue
            y = z - trk.x
            d2 = y.T @ S_inv @ y
            if d2 < best_d:
                best_d, best_j = d2, j

        if best_j is not None and best_d < GT_TH:
            z = measurements[best_j]
            K = trk.P @ H.T @ S_inv
            trk.x = trk.x + K @ (z - trk.x)
            trk.P = (np.eye(2) - K @ H) @ trk.P
            trk.misses = 0
            trk.hits += 1
            used.add(best_j)
    
    # Track Birth
    for j, z in enumerate(measurements):
        if j not in used:
            tracks.append(Track(z))
    # Track Death
    tracks = [t for t in tracks if t.misses < Max_miss]
    tracks_hist.append([t.x.copy() for t in tracks])        
# PLOT
print("Generating Diagnostic Animation...")


fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.suptitle(f"Processing Dashboard (Last Frame)", fontsize=16)

# A. Time Domain (Real part of 1st chirp)
line_sig, = axs[0, 0].plot(t_fast*1e6, hist_signal[0])
axs[0, 0].set_title("1. Beat Signal (Chirp 0)")
axs[0, 0].set_ylim(np.min(hist_signal)*1.2, np.max(hist_signal)*1.2)
axs[0, 0].set_xlabel("Time (us)")
axs[0, 0].grid(True)

# 2. Range Profile Plot
line_rng, = axs[0,1].plot(ranges, hist_range[0])
axs[0,1].set_title("2. Range Profile")
axs[0,1].set_ylim(0, 100) # Assuming ~max dB
axs[0,1].set_xlabel("Range (m)")
axs[0,1].grid(True)

# 3. Range-Doppler Map (Heatmap)
extent = [ranges[0], ranges[-1], vel_axis[0], vel_axis[-1]]
im_rd = axs[1,0].imshow(hist_rd_map[0], aspect='auto', origin='lower', extent=extent, cmap='jet', vmin=40, vmax=110)
axs[1,0].set_title("3. Range-Doppler Heatmap")
axs[1,0].set_xlabel("Range (m)")
axs[1,0].set_ylabel("Velocity (m/s)")

# 4. Final Tracking (Overlaid on CFAR Mask)
# We plot the CFAR mask in gray, and the TRACKS as red dots on top
im_cfar = axs[1,1].imshow(hist_cfar[0], aspect='auto', origin='lower', extent=extent, cmap='gray_r')
scat_tracks = axs[1,1].scatter([], [], c='red', s=100, label='Tracker Output')
axs[1,1].set_title("4. CFAR Mask + Tracker Output")
axs[1,1].set_xlabel("Range (m)")
axs[1,1].legend(loc='upper right')

def update(frame):
    # Update Plot 1
    line_sig.set_ydata(hist_signal[frame])
    
    # Update Plot 2
    line_rng.set_ydata(hist_range[frame])
    
    # Update Plot 3
    im_rd.set_data(hist_rd_map[frame])
    
    # Update Plot 4
    im_cfar.set_data(hist_cfar[frame])
    
    # Update Tracks
    pts = np.array(tracks_hist[frame])
    if len(pts) > 0:
        scat_tracks.set_offsets(pts)
    else:
        scat_tracks.set_offsets(np.empty((0,2)))
        
    return line_sig, line_rng, im_rd, im_cfar, scat_tracks

ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
ani.save("FMCW_Dashboard.gif", writer="pillow", dpi=120)
plt.close()
print("Done! Saved 'FMCW_Dashboard.gif'")
#fig, ax = plt.subplots(figsize=(8,5))
#ax.set_xlim(45, 70)
#ax.set_ylim(-30, 5)
#ax.set_xlabel("Range(m)")
#ax.set_ylabel("Velocity(m/s)")
#ax.grid()
#ax.set_title("FMCW Multi- Target Tracking")


#scat = ax.scatter([], [], s=100, c='red', marker='o', label='Tracked Target')
#ax.legend(loc='upper right')

#def update (i):
    #pts = np.array(tracks_hist[i])
    #scat.set_offsets(pts if len(pts) else np.empty((0,2)))
    #ax.set_title(f"Frame {i} | Active Tracks: {len(pts)}")
    #if len(pts) > 0:
     #   scat.set_offsets(pts)
    #else:
      #  scat.set_offset(np.empty((0,2)))    
    #return scat,
#ani = FuncAnimation(fig, update, frames=len(tracks_hist), interval=80)
#ani.save("FMCW_multi_traget_tracking.gif", writer="pillow", dpi=200)
#plt.close()
#print("Saved: FMCW_multi_traget_tracking.gif")

