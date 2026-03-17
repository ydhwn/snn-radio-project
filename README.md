# Neuromorphic Automatic Modulation Classification (AMC) for Cognitive Radio

**Major Project: B.Tech Electronics & Telecommunication (Final Year)**

## 🚀 Project Overview
This project implements an **Energy-Efficient Spiking Neural Network (SNN)** to classify 6 different modulation schemes (BPSK, QPSK, 8PSK, 16QAM, 16PSK, 64QAM) from noisy radio signals. It is designed for **low-power spectrum sensing** in next-generation wireless networks (5G/6G/Cognitive Radio).

---

## 🛠️ Key Features
1.  **Signal Generation (DSP)**: Synthetic generation of IQ samples with AWGN, RRC pulse shaping, and matched filtering.
2.  **Neuromorphic AI**: Uses **Leaky Integrate-and-Fire (LIF)** neurons from the `snntorch` library for biologically-inspired, low-power inference.
3.  **Channel Impairments**: Robustness testing against **Carrier Frequency Offset (CFO)** and **Rayleigh Fading**.
4.  **Hardware-Aware Profiling**: Estimates theoretical energy consumption in nano-Joules (nJ) per classification.
5.  **Industry Deployment**: Supports **ONNX** and **TorchScript** export for edge-device integration.
6.  **Interactive Demo**: Real-time Streamlit dashboard for signal visualization and classification.

---

## 📂 Project Structure
- `src/app.py`: **Live Demo Dashboard (Streamlit)**.
- `src/train.py`: Training pipeline with **SNR Curriculum Learning**.
- `src/inference.py`: Standalone inference engine for production backends (ONNX/TS).
- `src/export.py`: Model export utility for deployment.
- `src/signal_generator.py`: DSP module for generating modulated waveforms.
- `src/snn_model.py`: Spiking Neural Network architecture.
- `reports/`: Contains all generated plots, confusion matrices, and the **Presentation Pack**.
- `submission/`: Consolidated folder for final project submission.

---

## 🏃 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run the Live Demo (Highly Recommended for Presentation)
```bash
streamlit run src/app.py
```
*This opens a browser-based dashboard where you can adjust SNR, CFO, and Fading to see the SNN predict in real-time.*

### 3. Training the Model
```bash
python -m src.cli --samples 1200 --symbols 4096 --batch 128 --presentation
```

### 4. Export & Benchmark
```bash
python -m src.cli --export
python -m src.cli --benchmark deploy/model.onnx
```

### 🌍 Deploying as a Public App (Production)
To move from `localhost` to an "actual app" accessible anywhere:

**A. Streamlit Cloud (Free & Recommended)**
1.  Push this code to a **GitHub** repository.
2.  Log in to [Streamlit Cloud](https://share.streamlit.io/).
3.  Click "New app" and select your repo, branch, and `src/app.py`.
4.  Your app is now live at `https://your-project.streamlit.app`!

**B. Docker Container (Industry Standard)**
If you have Docker installed, you can build and run a professional container:
```bash
# Build the image
docker build -t snn-radio-app .

# Run the app
docker run -p 8501:8501 snn-radio-app
```

---

## 📄 Documentation
-   **[Final Project Report](README_FINAL.md)**: A detailed report in IEEE format for your project documentation.
-   **[Presentation Pack](reports/presentation_pack.pdf)**: A PDF containing all key figures and results for your supervisor.

---

## 🎓 Author
**[Your Name]**  
Final Year Student, Electronics & Telecommunication  
[Your University]
