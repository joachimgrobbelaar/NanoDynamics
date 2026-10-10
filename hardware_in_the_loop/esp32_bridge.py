import serial
import struct
import time
import numpy as np

class ESP32HardwareAgent:
    """
    Hardware-In-The-Loop (HIL) Bridge to offload PINN neural network 
    inference to the physical ESP32 microcontroller over USB.
    """
    def __init__(self, port='/dev/ttyUSB0', baudrate=115200):
        print(f"Connecting to ESP32 on {port}...")
        self.ser = serial.Serial(port, baudrate, timeout=2)
        
        # Reset the ESP32 using DTR/RTS to restart the sketch
        self.ser.setDTR(False)
        self.ser.setRTS(False)
        time.sleep(0.1)
        self.ser.setDTR(True)
        self.ser.setRTS(True)
        
        print("Waiting for ESP32 TinyML initialization...")
        # Wait for the firmware to send the READY signal
        while True:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if "READY" in line:
                print("ESP32 TinyML Agent is READY!")
                break
            time.sleep(0.01)
            
    def step(self, state_vector):
        """
        Sends the 6D state vector (x, y, z, vx, vy, vz) to the ESP32,
        waits for the TinyML model to run inference, and returns the result.
        """
        # Ensure it's exactly 6 floats
        if len(state_vector) != 6:
            raise ValueError("State vector must have 6 elements")
            
        # Pack the 6 floats into a 24-byte binary payload (little-endian)
        payload = struct.pack('<6f', *state_vector)
        
        # Send to ESP32
        self.ser.write(payload)
        self.ser.flush()
        
        # Read the 24-byte response (6 floats)
        response = self.ser.read(24)
        if len(response) == 24:
            next_state = struct.unpack('<6f', response)
            return np.array(next_state, dtype=np.float32)
        else:
            raise RuntimeError(f"ESP32 timeout or partial data: received {len(response)} bytes")

    def close(self):
        self.ser.close()

if __name__ == "__main__":
    # Test the bridge locally
    agent = ESP32HardwareAgent()
    
    test_state = [1000.0, 2000.0, 3000.0, 7.5, 0.0, 0.0]
    print(f"Sending test state: {test_state}")
    
    start_time = time.time()
    next_state = agent.step(test_state)
    latency = (time.time() - start_time) * 1000
    
    print(f"Received next state: {next_state}")
    print(f"Hardware Inference Latency: {latency:.2f} ms")
    agent.close()
