# FMCW Radar Perception & Tracking System

This project implements a **complete FMCW radar perception pipeline**
covering **digital signal processing, detection, and multi-frame target tracking**.

The system processes FMCW radar signals to detect **multiple targets**
and track them consistently over time using a **Cartesian Kalman filter
with full track lifecycle management**.

---

## ✅ Completed Features
covering **signal processing, detection, and multi-frame target tracking**.

The system progresses from raw FMCW signal processing to **stable,
multi-target tracking** using a **Cartesian Kalman filter with track lifecycle
management**.

---

## Completed Features

✔ FMCW radar DSP  
✔ Fast CFAR-based detection  
✔ Multi-target detection  
✔ Cartesian Kalman tracking  
✔ Track lifecycle management  

---

## System Overview

<<<<<<< HEAD
The project follows an end-to-end radar perception flow:
=======
The project processes FMCW radar signals through the following stages:
>>>>>>> 4d54087 (Updated Version)

- FMCW radar signal simulation
- Range FFT and Doppler FFT (DSP layer)
- Range–Doppler map generation
- **Fast CFAR detection** for multiple targets
<<<<<<< HEAD
- Conversion from polar to **Cartesian coordinates**
- **Kalman filter–based multi-frame tracking**
- Track initiation, confirmation, maintenance, and deletion
- Visualization via a dashboard-style output

This is a **full perception + tracking system**, not just isolated DSP blocks.
=======
- Conversion to **Cartesian coordinates**
- **Kalman filter–based multi-frame tracking**
- Track initialization, confirmation, maintenance, and deletion
- Visualization of detections and tracked targets

This represents a **full perception-and-tracking loop**, not just isolated algorithms.

---

## Project Structure

```text
fmcw/
├── src/
│   └── main.py
├── docs/
│   └── FMCW_Dashboard.gif
├── README.md
├── requirements.txt
└── .gitignore
