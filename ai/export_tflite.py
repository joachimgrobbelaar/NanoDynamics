import os
import sys
import numpy as np
import torch
import litert_torch

AI_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(AI_DIR)
for _path in (AI_DIR, PROJECT_ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from transition_model import TransitionMLP

def main():
    # 1. Load PyTorch model
    ckpt_path = os.path.join(AI_DIR, "checkpoints", "transition.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = ckpt["config"]["hidden_dim"]
    layers = ckpt["config"]["hidden_layers"]
    model = TransitionMLP(hidden_dim=hidden, hidden_layers=layers)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 2. Convert PyTorch to TFLite via litert-torch
    dummy_input = torch.zeros(1, 6)
    edge_model = litert_torch.convert(model, (dummy_input,))
    tflite_path = os.path.join(AI_DIR, "checkpoints", "transition.tflite")
    edge_model.export(tflite_path)
    print(f"Exported TFLite to {tflite_path}")

    # 3. Read the TFLite bytes
    with open(tflite_path, "rb") as f:
        tflite_model = f.read()

    # 4. Convert to C array for ESP32
    c_array_path = os.path.join(AI_DIR, "checkpoints", "transition_model_data.h")
    with open(c_array_path, "w") as f:
        f.write("#ifndef TRANSITION_MODEL_DATA_H\n")
        f.write("#define TRANSITION_MODEL_DATA_H\n\n")
        f.write("unsigned char transition_tflite[] = {\n")
        for i, val in enumerate(tflite_model):
            f.write(f"0x{val:02x}, ")
            if (i + 1) % 12 == 0:
                f.write("\n")
        f.write("\n};\n")
        f.write(f"unsigned int transition_tflite_len = {len(tflite_model)};\n\n")
        f.write("#endif // TRANSITION_MODEL_DATA_H\n")
    print(f"Exported C array to {c_array_path}")

if __name__ == "__main__":
    main()
