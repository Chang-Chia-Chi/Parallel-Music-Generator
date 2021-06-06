import asyncio
import json
import os
import logging
import sys
sys.path.append("../musicGen")
from musicGen import main as musicGen
from websockets import serve

WSPORT = os.getenv('WSPORT', 5566)

logging.basicConfig()

USERS = set()

async def relay(websocket, payload):
    # payload = json.loads(payload)
    print("payload:", payload)
    # TODO: call generate music model and send msg back
    if 'tune' in payload and payload["tune"] != "":
        music = musicGen(payload["tune"])
        await websocket.send(music)
    else:
        await websocket.send("unsupported payload. payload need to pass 'tune' param.")

async def broadcast(websocket, payload):
    if USERS:  # asyncio.wait doesn't accept an empty list
        await asyncio.wait([user.send(payload) for user in USERS])

async def register(websocket, payload):
    USERS.add(websocket)

async def unregister(websocket, payload):
    USERS.remove(websocket)

async def handler(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        if 'action' in data and data["action"] != "" :
            await globals()[f'{data["action"]}'](websocket, data["payload"])
        else:
            logging.error("unsupported action event, msg: %s", message)

async def main():
    async with serve(handler, '0.0.0.0', WSPORT):
        await asyncio.Future()  # run forever

asyncio.run(main())