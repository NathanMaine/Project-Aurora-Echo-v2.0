"""Test script for continuous recording WebSocket lifecycle.

Simulates: connect → start → stream audio → wait for periodic chunk → stop → final result.
Uses synthetic speech-like audio (sine wave) to trigger Whisper transcription.
"""
import asyncio
import json
import struct
import math
import sys
import time

import websockets


WS_URL = "ws://localhost:8000/ws"
SAMPLE_RATE = 16000
# Generate 2 seconds of 440Hz sine wave as 16-bit PCM to simulate speech
DURATION_SEC = 2
NUM_SAMPLES = SAMPLE_RATE * DURATION_SEC


def generate_sine_pcm(freq=440, duration=2, sample_rate=16000):
    """Generate PCM int16 bytes of a sine wave."""
    samples = []
    for i in range(sample_rate * duration):
        t = i / sample_rate
        val = int(32767 * 0.5 * math.sin(2 * math.pi * freq * t))
        samples.append(val)
    return struct.pack(f"<{len(samples)}h", *samples)


async def test_early_stop():
    """Test: connect, start, stream a little audio, stop before chunk interval."""
    print("\n=== TEST: Early Stop (before chunk interval) ===")
    async with websockets.connect(WS_URL) as ws:
        # Start
        await ws.send(json.dumps({"type": "start", "sampleRate": SAMPLE_RATE}))
        resp = json.loads(await ws.recv())
        print(f"  start response: {resp}")
        assert resp["type"] == "status" and resp["status"] == "recording", f"Expected recording status, got {resp}"

        # Stream some audio
        pcm = generate_sine_pcm(440, 3, SAMPLE_RATE)
        chunk_size = 8192
        for i in range(0, len(pcm), chunk_size):
            await ws.send(pcm[i:i+chunk_size])
        print(f"  sent {len(pcm)} bytes of audio")

        # Wait a couple seconds (less than CHUNK_INTERVAL=30s) then stop
        await asyncio.sleep(2)
        await ws.send(json.dumps({"type": "stop"}))
        print("  sent stop")

        # Collect all responses until final
        messages = []
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=60)
                msg = json.loads(raw)
                messages.append(msg)
                print(f"  received: type={msg.get('type')}, status={msg.get('status', '-')}")
                if msg["type"] == "final":
                    break
        except asyncio.TimeoutError:
            print("  TIMEOUT waiting for final message!")
            return False

        final = messages[-1]
        print(f"  final keys: {list(final.keys())}")
        if "error" in final and final["error"]:
            print(f"  final error: {final['error']}")
        else:
            transcript = final.get("transcription", "")
            summary = final.get("summary", "")
            actions = final.get("actions", [])
            print(f"  transcript length: {len(transcript)} chars")
            print(f"  summary length: {len(summary)} chars")
            print(f"  actions count: {len(actions)}")

    print("  PASSED")
    return True


async def test_no_audio():
    """Test: start then immediately stop with no audio."""
    print("\n=== TEST: No Audio (immediate stop) ===")
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"type": "start", "sampleRate": SAMPLE_RATE}))
        resp = json.loads(await ws.recv())
        print(f"  start response: {resp}")

        # Immediately stop with no audio
        await ws.send(json.dumps({"type": "stop"}))
        print("  sent stop (no audio)")

        messages = []
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=30)
                msg = json.loads(raw)
                messages.append(msg)
                print(f"  received: type={msg.get('type')}, status={msg.get('status', '-')}")
                if msg["type"] == "final":
                    break
        except asyncio.TimeoutError:
            print("  TIMEOUT waiting for final message!")
            return False

        final = messages[-1]
        # With no audio, we should get a final with empty or error
        print(f"  final: {json.dumps(final, indent=2)[:200]}")

    print("  PASSED")
    return True


async def test_disconnect():
    """Test: start recording, stream audio, then disconnect without stop."""
    print("\n=== TEST: Disconnect During Recording ===")
    try:
        async with websockets.connect(WS_URL) as ws:
            await ws.send(json.dumps({"type": "start", "sampleRate": SAMPLE_RATE}))
            resp = json.loads(await ws.recv())
            print(f"  start response: {resp}")

            pcm = generate_sine_pcm(440, 2, SAMPLE_RATE)
            await ws.send(pcm)
            print(f"  sent {len(pcm)} bytes, disconnecting without stop...")
            # Just close — don't send stop
    except Exception as e:
        print(f"  disconnect exception (expected): {e}")

    # Give server a moment to clean up
    await asyncio.sleep(2)

    # Verify server is still healthy
    async with websockets.connect(WS_URL) as ws:
        await ws.send(json.dumps({"type": "start", "sampleRate": SAMPLE_RATE}))
        resp = json.loads(await ws.recv())
        assert resp["type"] == "status", f"Server not healthy after disconnect: {resp}"
        await ws.send(json.dumps({"type": "stop"}))
        # Drain responses
        try:
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=15)
                msg = json.loads(raw)
                if msg["type"] == "final":
                    break
        except asyncio.TimeoutError:
            pass

    print("  Server healthy after disconnect - PASSED")
    return True


async def main():
    print("Aurora Echo — Continuous Recording Test Suite")
    print("=" * 50)
    results = {}

    for name, test_fn in [
        ("early_stop", test_early_stop),
        ("no_audio", test_no_audio),
        ("disconnect", test_disconnect),
    ]:
        try:
            results[name] = await test_fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            results[name] = False

    print("\n" + "=" * 50)
    print("RESULTS:")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
