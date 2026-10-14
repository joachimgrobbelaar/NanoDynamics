import asyncio
import json
import websockets
from esp32_bridge import ESP32HardwareAgent

async def run_worker():
    uri = "ws://localhost:8000/ws/hil-worker" 
    print(f"Connecting to {uri}...")
    
    agent = None
    try:
        agent = ESP32HardwareAgent(port='/dev/ttyUSB0', baudrate=115200)
        print("Hardware agent initialized.")
    except Exception as e:
        print(f"Could not initialize hardware agent: {e}")
        return

    async with websockets.connect(uri) as websocket:
        print("Connected to cloud backend as HIL worker. Waiting for simulation requests...")
        step_count = 0
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if "state" in data:
                    state = data["state"]
                    next_state = agent.step(state)
                    await websocket.send(json.dumps({"state": next_state.tolist()}))
                    
                    step_count += 1
                    if step_count % 50 == 0:
                        print(f"[{step_count}] Inference steps executed on ESP32 Hardware")
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")
                break
            except Exception as e:
                print(f"Error processing message: {e}")

if __name__ == "__main__":
    asyncio.run(run_worker())
