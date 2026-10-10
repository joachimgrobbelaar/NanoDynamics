import asyncio
import json
import websockets
from esp32_bridge import ESP32HardwareAgent

async def run_worker():
    uri = "ws://localhost:8000/ws/hil-worker" # Localhost for now, user will change this to render URL
    print(f"Connecting to {uri}...")
    
    agent = None
    try:
        agent = ESP32HardwareAgent(port='/dev/ttyUSB0', baudrate=115200)
        print("Hardware agent initialized.")
    except Exception as e:
        print(f"Could not initialize hardware agent: {e}")
        return

    async with websockets.connect(uri) as websocket:
        print("Connected to cloud backend as HIL worker.")
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                if "state" in data:
                    state = data["state"]
                    # step() is synchronous, but it's fast over serial (6ms)
                    # we could run it in thread but it's fast enough
                    next_state = agent.step(state)
                    await websocket.send(json.dumps({"state": next_state.tolist()}))
            except websockets.exceptions.ConnectionClosed:
                print("Connection closed by server.")
                break
            except Exception as e:
                print(f"Error processing message: {e}")

if __name__ == "__main__":
    asyncio.run(run_worker())
