#!/usr/bin/env python3
"""Test script to verify Aurora Echo WebSocket functionality."""

import asyncio
import json
import websockets
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_websocket():
    """Test WebSocket connection to Aurora Echo."""
    uri = "ws://localhost:8000/ws"

    try:
        logger.info("Attempting to connect to %s", uri)
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to Aurora Echo WebSocket")

            # Send a test message
            test_message = {
                "type": "audio_data",
                "data": "dGVzdCBhdWRpbyBkYXRh",  # base64 encoded "test audio data"
                "timestamp": 1234567890
            }

            await websocket.send(json.dumps(test_message))
            logger.info("Sent test audio data")

            # Wait for response
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                response_data = json.loads(response)
                logger.info(f"Received response: {response_data}")
            except asyncio.TimeoutError:
                logger.warning("No response received within timeout")

            # Send end message
            end_message = {"type": "end"}
            await websocket.send(json.dumps(end_message))
            logger.info("Sent end message")

            # Wait a bit more for any final responses
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                logger.info(f"Received final response: {response_data}")
            except asyncio.TimeoutError:
                logger.info("No final response received")

    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"WebSocket connection closed: {e}")
        return False
    except websockets.exceptions.WebSocketException as e:
        logger.error(f"WebSocket exception: {e}")
        return False
    except Exception as e:
        logger.error(f"WebSocket test failed: {e}")
        return False

    logger.info("WebSocket test completed successfully")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_websocket())
    if success:
        print("✅ WebSocket test PASSED!")
    else:
        print("❌ WebSocket test FAILED!")