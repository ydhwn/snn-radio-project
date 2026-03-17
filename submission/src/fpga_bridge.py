import serial
import time
import numpy as np

# -----------------------------------------------------------------------------
# FPGA-Python Data Bridge (Hardware-in-the-Loop)
# -----------------------------------------------------------------------------
# This script feeds real IQ data from the SNN-Radio App to your FPGA board
# via UART/Serial. Ensure your FPGA UART is set to 115200 baud.
# -----------------------------------------------------------------------------

def send_iq_to_fpga(iq_samples, port='COM3', baud=115200):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        print(f"--- FPGA Data Bridge Active on {port} ---")
        
        # Prepare fixed-point IQ data (16-bit)
        # Scaled to avoid overflow in FPGA fixed-point logic
        i_fixed = (np.real(iq_samples) * 4096).astype(np.int16)
        q_fixed = (np.imag(iq_samples) * 4096).astype(np.int16)
        
        # Interleave I and Q for serial transmission
        data_stream = np.empty((i_fixed.size + q_fixed.size,), dtype=np.int16)
        data_stream[0::2] = i_fixed
        data_stream[1::2] = q_fixed
        
        # Send byte-by-byte
        print(f"Sending {len(iq_samples)} samples to FPGA...")
        ser.write(data_stream.tobytes())
        
        # Wait for FPGA response (Class Index)
        response = ser.read(1)
        if response:
            print(f"FPGA Classified Result: {int.from_bytes(response, 'big')}")
        
        ser.close()
    except Exception as e:
        print(f"Error connecting to FPGA: {e}")
        print("Note: Ensure your FPGA board is connected and COM port is correct.")

if __name__ == "__main__":
    # Test with dummy data
    dummy_iq = np.random.randn(512) + 1j*np.random.randn(512)
    send_iq_to_fpga(dummy_iq)
