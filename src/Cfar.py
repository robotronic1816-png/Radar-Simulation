import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec 
from scipy.signal import windows, butter, filtfilt
from sklearn.cluster import DBSCAN
from scipy.ndimage import convolve
from scipy.optimize import linear_sum_assignment

def cfar_2d_fast(rd_db, tr=12, gr=6, td=8, gd=4, offset_db=10.0):
    d_win = 2 * (td + gd) + 1
    r_win = 2 * (tr + gr) + 1
    kernel = np.ones((d_win, r_win))
    center_d = td + gd
    center_r = tr + gr
    kernel[center_d - gd : center_d + gd + 1, center_r - gr : center_r + gr + 1] = 0
    # Normalize
    kernel /= np.sum(kernel)
    # Convolution
    rd_lin = 10**(rd_db / 10.0)
    noise_lin = convolve(rd_lin, kernel, mode="reflect")
    noise_db = 10 * np.log10(noise_lin + 1e-9)
    return rd_db > (noise_db + offset_db)
# MUSIC AOA
def music_aoa_multisnap(X, lam, d_ant, n_src=1,
                        angle_grid=np.linspace(-90, 90, 361),
                        diag_load=1e-3):
    """
    X : complex ndarray (N_rx, K snapshots)
    returns: theta_hat (rad), Music spectrum
    """

    N_rx, N_snap = X.shape

    # Spatial covariance
    R = (X @ X.conj().T) / N_snap
    R += diag_load * np.eye(N_rx)

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(R)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    # Noise subspace
    En = eigvecs[:, n_src:]

    # MUSIC spectrum
    P = np.zeros(len(angle_grid))
    for i, ang in enumerate(np.deg2rad(angle_grid)):
        a = np.exp(
            1j * 2 * np.pi * d_ant * np.arange(N_rx) * np.sin(ang) / lam
        ).reshape(-1, 1)
        den = np.real(a.conj().T @ En @ En.conj().T @ a).item()
        P[i] = 1.0 / (den + 1e-12)
        theta_hat = np.deg2rad(angle_grid[np.argmax(P)])
         
    return theta_hat, P


# Jacobian : Polar To Cartesian

def jacobian_polar_to_cart(R, v, theta):
    J = np.zeros((4,3))
    J[0,0] = np.cos(theta)
    J[0,2] = -R * np.sin(theta)
    J[1,0] = np.cos(theta)
    J[1,2] = -R * np.sin(theta)
    J[2,1] = np.cos(theta)
    J[2,2] = -v * np.sin(theta)
    J[3,1] = np.cos(theta)
    J[3,2] = v * np.sin(theta)
    return J

# IMM Track Class(cartesion)
class IMMTrack:
    _id = 0
    def __init__(self, z, R_cart, dt):
        self.id = IMMTrack._id; IMMTrack._id += 1
        self.dt = dt
# CV
        self.x_cv =z.reshape(4, 1)
        self.P_cv = R_cart.copy()
# CA
        self.x_ca = np.vstack([z.reshape(4,1), np.zeros((2,1))])
        P_ca_init = np.eye(6) * 10
        P_ca_init[:4, :4] = R_cart.copy()
        self.P_ca = P_ca_init
# Model Probab
        self.mu = np.array([0.5,0.5]) 

# Markov
        self.PI = np.array([[0.95, 0.05],
                            [0.05, 0.95]])
        self.age = 1
        self.hits = 1
        self.misses = 0
        self.state = "tentative"
    def mix(self):
        mu_pred = self.PI.T @ self.mu

        w = np.zeros((2,2))
        for j in range (2):
            for i in range(2):
                w[i,j] = self.PI[i,j] * self.mu[i] / (mu_pred[j] + 1e-9)

        x0_cv = w[0,0]*self.x_cv + w[1,0]*self.x_ca[:4]
# Mix Covariance
        diff_cv = self.x_cv - x0_cv
        diff_ca_cv = self.x_ca[:4] - x0_cv       
        P0_cv = (
            w[0,0]*(self.P_cv + diff_cv @ diff_cv.T) + \
            w[1,0]*(self.P_ca[:4,:4] + diff_ca_cv @ diff_ca_cv.T)
        )
        x0_ca = np.vstack([
            w[0,1]*self.x_cv + w[1,1]*self.x_ca[:4],
            np.zeros((2,1))
        ])
        P0_ca = self.P_ca.copy()
        diff_cv_ca = self.x_cv - x0_ca[:4]
        diff_ca = self.x_ca[:4] - x0_ca[:4]
        P0_ca[:4,:4] = (
            w[0,1]*(self.P_cv + diff_cv_ca @ diff_cv_ca.T) + \
            w[1,1]*(self.P_ca[:4,:4] + diff_ca @ diff_ca.T)
        )
        self.x_cv, self.P_cv = x0_cv, P0_cv
        self.x_ca, self.P_ca = x0_ca, P0_ca
        self.mu = mu_pred        

    def predict(self):
        self.mix()
        dt = self.dt

        F_cv = np.array([[1,0,dt,0],
                        [0,1,0,dt],
                        [0,0,1,0],
                        [0,0,0,1]])
        Q_cv = np.eye(4) * 0.3

        F_ca = np.array([
            [1,0,dt,0,0.5*dt**2,0],
            [0,1,0,dt,0,0.5*dt**2],
            [0,0,1,0,dt,0],
            [0,0,0,1,0,dt],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1]
        ])
        Q_ca = np.eye(6) * 0.5

        self.x_cv = F_cv @ self.x_cv
        self.P_cv = F_cv @ self.P_cv @ F_cv.T + Q_cv
        self.x_ca = F_ca @ self.x_ca
        self.P_ca = F_ca @ self.P_ca @ F_ca.T + Q_ca

    def update(self, z, R_cart):

        H_cv = np.eye(4)
        y_cv = z.reshape(4,1) - H_cv @ self.x_cv
        S_cv = H_cv @ self.P_cv @ H_cv.T + R_cart
        K_cv = self.P_cv @ H_cv.T @ np.linalg.inv(S_cv)
        self.x_cv += K_cv @ y_cv
        self.P_cv = (np.eye(4) - K_cv @ H_cv) @ self.P_cv

        det_S_cv = np.linalg.det(S_cv)
        norm_const_cv = 1.0 / (np.sqrt((2*np.pi)**4 * det_S_cv) + 1e-9)
        L_cv = norm_const_cv * np.exp(-0.5 * y_cv.T @ np.linalg.inv(S_cv) @ y_cv)

        H_ca = np.hstack([np.eye(4), np.zeros((4,2))])
        y_ca = z.reshape(4,1) - H_ca @ self.x_ca
        S_ca = H_ca @ self.P_ca @ H_ca.T + R_cart
        K_ca = self.P_ca @ H_ca.T @ np.linalg.inv(S_ca)
        self.x_ca += K_ca @ y_ca
        self.P_ca = (np.eye(6) - K_ca @ H_ca) @ self.P_ca

        det_S_ca = np.linalg.det(S_ca)
        norm_const_ca = 1.0 / (np.sqrt((2 *np.pi)**4 * det_S_ca) + 1e-9)
        L_ca = norm_const_ca * np.exp(-0.5 * y_ca.T @ np.linalg.inv(S_ca) @ y_ca)


        self.mu *= np.array([L_cv.item(), L_ca.item()])
        self.mu /= (np.sum(self.mu) + 1e-9)

        self.hits += 1
        self.misses = 0
        if self.state == "tentative" and self.hits >= 6:
            self.state = "confirmed"

    def miss(self):
        self.misses += 1
        self.age += 1

    def fused(self):
        x = self.mu[0]*self.x_cv + self.mu[1]*self.x_ca[:4]
        return x            

# RADAR PARAMETERS
c = 3e8
fc = 77e9
B = 200e6
T = 30e-6
Sf = B / T
fs = 20e6
lam = c / fc

num_chirps = 64
num_frames = 120
SNR_dB = 15 ; snr_lin = 10**(SNR_dB/10)
max_range = 150
N_rx = 4
d_ant = lam / 2
dt = num_chirps * T


targets = [
    {"Rn": 55.0, "Vl": -10, "theta": np.deg2rad(15)},
    {"Rn": 50.0, "Vl": -10, "theta": np.deg2rad(-15)}
    
]

print("FMCW radar simulation started")


# Signal Setup
t_fast = np.arange(0, T, 1/fs)
N_fast = len(t_fast)
t_grid, k_grid = np.meshgrid(t_fast, np.arange(num_chirps))

tracks =[]
tracks_hist = []         
meas_hist = []
hist_signal = []
hist_range = []
hist_rd_map = []
hist_cfar = []
imu_hist = {}
print("\nInitiating Multi- frame Tracking  ...\n")

# Frame Loop
for frame in tqdm(range(num_frames), desc="Processing"): 
    # FMCW Simulation
    beat = np.zeros((N_rx, num_chirps, N_fast), dtype=complex)
    
    for tgt in targets:
            R = tgt["Rn"] + tgt["Vl"] * (frame * dt + k_grid * T)
            fb = 2 * Sf * R / c
            fD = 2 * tgt["Vl"] * fc / c
            steering = np.exp(
                1j * 2 * np.pi * d_ant * np.arange(N_rx) * np.sin(tgt["theta"]) / lam
            )
            phase = 2 * np.pi * (fb * t_grid + fD * k_grid * T)
            for rx in range(N_rx):
                beat[rx] += steering[rx] * np.exp(1j * phase)

    noise = (np.random.randn(*beat.shape) + 1j * np.random.randn(*beat.shape))    
    beat += noise * 10**(-SNR_dB / 20)
    beat *= windows.hann((N_fast))[None,None,:]
    hist_signal.append(np.real(beat[0, 0, :]))
# RANGE FFT
    rng_fft = np.fft.fft(beat, axis=2)[:,:,:N_fast//2]
    freqs = np.fft.fftfreq(N_fast, d=1/fs)[:N_fast//2]
    ranges = (c * freqs) / (2 * Sf)
    mask = ranges <= max_range
    rng_fft = rng_fft[:,:,mask]
    ranges_valid = ranges[mask]
    
    rng_profile = np.mean(np.abs(rng_fft[0]), axis=0)
    hist_range.append(20 * np.log10(rng_profile + 1e-6)) 

   # rng_fft -= np.mean(rng_fft, axis=0)
   # rng_fft *= windows.blackman(num_chirps)[:, None]
    #rng_fft = rng_fft / (np.linalg.norm(rng_fft, axis=0, keepdims=True) + 1e-6) 
# DOPPLER FFT
    #rng_fft -= np.mean(rng_fft, axis=1, keepdims=True)
    #rng_fft *= windows.blackman(num_chirps)[None,:, None]
    doppler_fft = np.fft.fftshift(
        np.fft.fft(rng_fft, axis=1),
        axes=1
)

    rd = np.abs(doppler_fft).mean(axis=0)
    rd_db = 20 * np.log10(rd + 1e-6)
    hist_rd_map.append(rd_db)
# CFAR
    
    det = cfar_2d_fast(rd_db)
    d_idx, r_idx = np.where(det)
    hist_cfar.append(det)
    vel_axis = np.fft.fftshift(np.fft.fftfreq(num_chirps, T)) * c / (2 * fc)

    measurements = []
    covs = []
    if len(r_idx) > 0:
       pts = np.column_stack((ranges_valid[r_idx]/10, vel_axis[d_idx]/5))
       labels = DBSCAN(eps=0.8, min_samples=3).fit(pts).labels_

       for lab in set(labels):
           if lab == -1:
               continue
           mask = labels == lab

           dd = d_idx[mask]
           rr = r_idx[mask]

           pwr = rd_db[dd, rr]
           idx = np.argmax(pwr)

           d0 , r0 = dd[idx], rr[idx]
           K = 5
           d_lo = max(0, d0 - K//2)
           d_hi = min(num_chirps, d0 + K//2 + 1)
           X = doppler_fft[:, :, r0]
           theta, spectrum = music_aoa_multisnap(X, lam, d_ant)
           Rm = ranges_valid[r0]
           Vm = vel_axis[d0]

           z = np.array([Rm*np.cos(theta),Rm*np.sin(theta),
                         Vm*np.cos(theta),Vm*np.sin(theta)])
           sigma_R = 0.15 + 0.002*Rm
           sigma_v = 0.2
           sigma_theta = np.deg2rad(2)/np.sqrt(snr_lin)

           R_polar = np.diag([sigma_R**2,sigma_v**2,sigma_theta**2])
           J = jacobian_polar_to_cart(Rm,Vm,theta)
           R_cart = J @ R_polar @ J.T

           measurements.append(z)
           covs.append(R_cart)                
           
    meas_hist.append(np.array(measurements)[:, :2] if measurements else np.empty((0, 2)))      
       
   
# Multi traget tracking (GNN + IMM)
        
    for t in tracks:
        t.predict()
        #trk.miss()

    nT = len(tracks)
    nM = len(measurements)
    asi_tracks = set()
    asi_meas = set()
    


    if nT and nM:
        C = np.full((nT, nM), 1e6)
        #R_val = np.diag([1, 5, 1, 5])
        #H = np.eye(4)

        for i, t in enumerate(tracks):
            x = t.fused()
           # P_fused = trk.mu[0]*trk.P_cv + trk.mu[1]*trk.P_ca[:4,:4]
            #S = H @ P_fused @ H.T + R_val
            #S_inv = np.linalg.inv(S)

            for j, z in enumerate(measurements):
                y = z.reshape(4,1) - x
                S = t.P_cv + covs[j]
                C[i, j] = (y.T @ np.linalg.inv(S) @ y).item()

        ri, ci = linear_sum_assignment(C)
        GATE = 25
        for i, j in zip(ri, ci):
            if C[i, j] < GATE:
                tracks[i].update(measurements[j], covs[j])
                asi_tracks.add(i)
                asi_meas.add(j)
        
    for i, t in enumerate(tracks):
        if i not in asi_tracks:
           t.miss()
    # Track Birth
    for j, z in enumerate(measurements):
        if j not in asi_meas:
         tracks.append(IMMTrack(z, covs[j], dt))
    
    # Track Death
    tracks = [t for t in tracks if t.misses < 10]
    
    tracks_hist.append([t.fused()[:2].flatten() for t in tracks if t.state == "confirmed"])
    meas_hist.append(np.array(measurements)[:, :2] if measurements else np.empty((0,2)))
    for t in tracks:
        if t.state == "confirmed":
            imu_hist.setdefault(t.id, []).append((frame, t.mu.copy()))
        
            


# PLOT
print("Generating Diagnostic Animation...")


fig = plt.figure(figsize=(14, 12))
gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 0.8])

ax_sig = fig.add_subplot(gs[0, 0])
ax_rng = fig.add_subplot(gs[0, 1])
ax_rd = fig.add_subplot(gs[1, 0])
ax_map = fig.add_subplot(gs[1, 1])
ax_imm = fig.add_subplot(gs[2, :])
plt.suptitle(f"FMCW MIMO (IMM + MUSIC)", fontsize=16)

# A. Beat Signal
line_sig, = ax_sig.plot(t_fast*1e6, hist_signal[0])
ax_sig.set_title("1. Beat Signal (Chirp 0)")
ax_sig.set_xlabel("Time (us)")
ax_sig.grid(True)
ax_sig.set_ylim(np.min(hist_signal)*1.2, np.max(hist_signal)*1.2)


# 2. Range Profile Plot
line_rng, = ax_rng.plot(ranges_valid, hist_range[0])
ax_rng.set_title("2. Range Profile")
ax_rng.set_ylim(0, 100) # Assuming ~max dB
ax_rng.set_xlabel("Range (m)")
ax_rng.grid(True)

# 3. Range-Doppler Map (Heatmap)
extent = [ranges_valid[0], ranges_valid[-1], vel_axis[0], vel_axis[-1]]
im_rd = ax_rd.imshow(hist_rd_map[0], aspect='auto', origin='lower', extent=extent, cmap='jet', vmin=40, vmax=110)
ax_rd.set_title("3. Range-Doppler Heatmap")
ax_rd.set_xlabel("Range (m)")
ax_rd.set_ylabel("Velocity (m/s)")

# 4. Final Tracking (Overlaid on CFAR Mask)
scat_meas = ax_map.scatter([], [], c='lime', marker='x', s=80, label='Meas')
scat_tracks = ax_map.scatter([], [], c='red', s=100, label='Track')
ax_map.set_title("4. Cartesian Map (Top-Down)")
ax_map.grid(True)
ax_map.set_xlabel("X  (m)")
ax_map.set_ylabel("Y  (m)")
ax_map.set_xlim(40, 75)
ax_map.set_ylim(-30, 30)
ax_map.legend()

# IMM Plot
ax_imm.set_title("5. IMM Mdoel Probability (Solid=CV, Dashed=CA)")
ax_imm.set_xlabel("frame"); ax_imm.set_ylabel("Probability")
ax_imm.set_ylim(0, 1.1); ax_imm.set_xlim(0, num_frames)
ax_imm.grid(True)

imm_lines ={}
colors = plt.cm.tab10.colors


def update(frame):
    
    line_sig.set_ydata(hist_signal[frame])    
    line_rng.set_ydata(hist_range[frame])
    im_rd.set_data(hist_rd_map[frame])
    pts_m = np.array(meas_hist[frame])
    pts_t = np.array(tracks_hist[frame])
    scat_meas.set_offsets(pts_m if len(pts_m) else np.empty((0,2)))
    scat_tracks.set_offsets(pts_t if len(pts_t) else np.empty((0,2)))
    for tid, history in imu_hist.items():
        data_now = [h for h in history if h[0] <= frame]

        if not data_now: continue

        if tid not in imm_lines:
            c = colors[tid % len(colors)]
            ln_cv, = ax_imm.plot([], [], '-', color=c, lw=1.5, label=f'T={tid} CV')
            ln_ca, = ax_imm.plot([], [], '--', color=c, lw=1.5, alpha=0.6)
            imm_lines[tid] = (ln_cv, ln_ca)

        frames = [h[0] for h in data_now]
        cv_probs = [h[1][0] for h in data_now]
        ca_probs = [h[1][1] for h in data_now]

        ln_cv, ln_ca = imm_lines[tid]
        ln_cv.set_data(frames, cv_probs)
        ln_ca.set_data(frames, ca_probs)    

    return line_sig, line_rng, im_rd, scat_tracks, scat_meas, *[ln for pair in imm_lines.values() for ln in pair]


ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)

with tqdm(total=num_frames, desc="Rendering GIF", unit="frame") as bar:
    def update_progress(current_frame, total_frames):
        bar.update(1) # Advance bar by 1 frame
        
    ani.save("FMCW_Dashboard IMM AOA.gif", writer="pillow", dpi=120, progress_callback = update_progress)
plt.close()
print("Done! Saved 'FMCW_Dashboard IMM AOA Final.gif'")
