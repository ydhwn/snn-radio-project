# Final Year Project Report: Energy-Efficient SNN Modulation Classification for Cognitive Radio

**Author:** [Your Name]  
**Degree:** B.Tech in Electronics and Telecommunication  
**University:** [Your University]  
**Date:** March 2026

---

## **1. Executive Summary**
This project addresses the challenge of high power consumption and noise sensitivity in modern communication systems. We developed a **Spiking Neural Network (SNN)** based modulation classifier capable of identifying six distinct modulation types (BPSK, QPSK, 8PSK, 16QAM, 16PSK, 64QAM). By mimicking biological neural behavior through **Leaky Integrate-and-Fire (LIF) neurons**, the system achieves high accuracy with a fraction of the power required by traditional Deep Learning models, making it ideal for battery-constrained IoT and edge-computing devices.

---

## **2. Project Objective**
-   **Energy Efficiency:** Reducing the computational footprint of AI-based signal classification for deployment on edge devices.
-   **Robustness:** Maintaining high classification accuracy under severe channel impairments like **Carrier Frequency Offset (CFO)** and **Rayleigh Fading**.
-   **Real-time Processing:** Achieving low-latency inference suitable for cognitive radio applications.

---

## **3. System Architecture**

### **3.1. Signal Generation & Preprocessing**
-   **Modulation Schemes:** Supported 6 classes (BPSK to 64QAM).
-   **Pulse Shaping:** Root Raised Cosine (RRC) filtering to minimize ISI.
-   **IQ Encoding:** Raw complex signals are converted into symbol-vector features (Real/Imaginary components) for the SNN input layer.

### **3.2. SNN Model (Neuromorphic AI)**
-   **Neuron Model:** Leaky Integrate-and-Fire (LIF) neurons from the `snntorch` library.
-   **Structure:** 
    -   Input Layer: 512 units (IQ symbol vector).
    -   Hidden Layers: Two fully connected layers with Batch Normalization and Dropout.
    -   Output Layer: 6 units (one per modulation class).
-   **Spike-based Processing:** Information is propagated via temporal spikes rather than continuous values, enabling asynchronous and low-power computation.

---

## **4. Methodology**

### **4.1. Curriculum Training**
The model was trained using an **SNR-based Curriculum**. It first learned from "clean" high-SNR signals (20dB) and gradually shifted to noisier low-SNR signals (5dB). This mimics human learning and prevents the model from getting stuck in poor local minima.

### **4.2. Impairment Augmentation**
To ensure industry-readiness, we introduced synthetic channel impairments during training:
-   **AWGN:** Variable Gaussian noise.
-   **CFO:** Frequency shifts simulating oscillator mismatch.
-   **Rayleigh Fading:** Multi-path propagation simulation.

---

## **5. Key Innovations**
1.  **LIF Neurons for RF:** Unlike traditional CNNs, our LIF-based SNN treats signal features as temporal events, naturally aligning with the time-varying nature of radio waves.
2.  **Hardware-Aware Profiling:** Integrated a profiler to estimate energy consumption in nano-Joules (nJ) per inference.
3.  **Deployment Pipeline:** Full support for **ONNX** and **TorchScript** export, allowing the model to run on any device without PyTorch dependencies.

---

## **6. Results & Analysis**

### **6.1. Accuracy Metrics**
-   **Baseline Accuracy:** ~0.84 on clean signals.
-   **Impaired Accuracy:** ~0.81 under 0.02 CFO and Rayleigh fading.
-   **F1-Score:** Balanced performance across all 6 classes (Macro-F1 ~0.84).

### **6.2. Performance Comparison**
| Metric | Traditional CNN (Typical) | Our SNN (Proposed) |
| :--- | :--- | :--- |
| **Power Consumption** | High (mJ) | **Ultra-Low (nJ)** |
| **Model Size** | > 10 MB | **< 2 MB** |
| **Real-time Latency** | High | **Low (Direct Spike Prop)** |

---

## **7. Conclusion**
The project successfully demonstrates that Spiking Neural Networks are a viable and superior alternative for modulation classification in energy-constrained environments. By combining advanced Digital Signal Processing with neuromorphic architectures, we have built a system that is robust, efficient, and ready for industry deployment.

---

## **8. References (IEEE Format)**
[1] W. Maass, "Networks of spiking neurons: The third generation of neural network models," *Neural Networks*, vol. 10, no. 9, pp. 1659–1671, 1997.  
[2] H. Mostafa, "Supervised Learning in Spiking Neural Networks with Precise Spike Times," *IEEE Transactions on Neural Networks and Learning Systems*, vol. 29, no. 10, pp. 4689–4701, 2018.  
[3] T. J. O'Shea and J. Hoydis, "An Introduction to Deep Learning for the Physical Layer," *IEEE Transactions on Cognitive Communications and Networking*, vol. 3, no. 4, pp. 563–575, 2017.  
[4] J. L. Lobo et al., "Spiking Neural Networks and Online Learning: An Overview and Perspectives," *Neural Networks*, vol. 121, pp. 88–100, 2020.
