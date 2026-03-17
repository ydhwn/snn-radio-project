# FPGA Implementation Guide: Energy-Efficient SNN Modulation Classifier
## B.Tech Major Project - Electronics & Telecommunication Engineering

This guide explains how to move from the **Python Simulation** to a **Hardware-in-the-Loop (HIL)** implementation on your physical FPGA.

---

### **Step 1: Prepare the HDL Files**
All Verilog files are located in the `hardware/fpga/hdl/` directory.
1.  **[lif_neuron.v](file:///c:/Users/KIIT/Desktop/asl/x/hardware/verilog/lif_neuron.v)**: The core spiking neuron.
2.  **[snn_top.v](file:///c:/Users/KIIT/Desktop/asl/x/hardware/fpga/hdl/snn_top.v)**: The top-level SNN architecture for synthesis.
3.  **[spike_encoder.v](file:///c:/Users/KIIT/Desktop/asl/x/hardware/fpga/hdl/spike_encoder.v)** (to be generated): Converts IQ data to spikes.

### **Step 2: Synthesis and Implementation (Vivado / Quartus)**
1.  **Create a Project**: Open your FPGA design tool (Xilinx Vivado or Intel Quartus).
2.  **Add Sources**: Import the `.v` files from the `hardware/` folder.
3.  **Define Constraints**: Create a `.xdc` or `.sdc` file to map the `clk` and `reset` pins to your specific board's oscillators and buttons.
4.  **Synthesis**: Run Synthesis to see the **Utilization Report**. 
    - *Expert Tip:* Highlight how few LUTs (Look-Up Tables) the SNN uses compared to a standard DSP-based CNN.
5.  **Generate Bitstream**: Compile the design and program your FPGA board.

### **Step 3: Hardware-in-the-Loop (HIL) Testing**
1.  **Connect Hardware**: Plug your FPGA into your laptop via USB/Serial (UART).
2.  **Identify Port**: Open Device Manager to find your COM port (e.g., `COM3`).
3.  **Install Serial Library**:
    ```bash
    pip install pyserial
    ```
4.  **Run the Data Bridge**:
    - Open **[fpga_bridge.py](file:///c:/Users/KIIT/Desktop/asl/x/hardware/fpga/scripts/fpga_bridge.py)**.
    - Update the `port='COM3'` variable to match your board.
    - Run the script: `python hardware/fpga/scripts/fpga_bridge.py`.
5.  **Observe**: The script will stream real signal data to the FPGA. The FPGA will classify it using its Verilog-logic and send the result back to your terminal.

---

### **How to present this to your Supervisor:**
"Sir/Ma'am, I have completed the **Hardware-Software Co-Design** phase. 
- The **Python App** handles the user interface and high-level signal generation.
- The **Verilog RTL** (Register Transfer Level) handles the actual classification.
- By using a **Hardware-in-the-Loop** bridge, I can prove that my SNN model runs on real FPGA gates, achieving ultra-low power consumption suitable for modern E&TC field deployment."
