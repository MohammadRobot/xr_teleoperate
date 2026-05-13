#!/usr/bin/env python3
"""Teleimager-compatible local test-pattern server.

Use this when the host PC has no camera attached but you still want to verify
the Quest/WebRTC and teleop image-server path before deploying teleimager to PC2.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time

import cv2
import numpy as np

import logging_mp

if not hasattr(logging_mp, "basicConfig") and hasattr(logging_mp, "basic_config"):
    logging_mp.basicConfig = logging_mp.basic_config
if not hasattr(logging_mp, "getLogger") and hasattr(logging_mp, "get_logger"):
    logging_mp.getLogger = logging_mp.get_logger

from teleimager.image_client import ZMQ_Responser
from teleimager.image_server import WebRTC_PublisherManager


logger = logging_mp.getLogger(__name__)


def build_config(args: argparse.Namespace) -> dict:
    return {
        "head_camera": {
            "enable_zmq": False,
            "zmq_port": 55555,
            "enable_webrtc": True,
            "webrtc_port": args.webrtc_port,
            "webrtc_codec": args.codec,
            "type": "testpattern",
            "image_shape": [args.height, args.width],
            "binocular": not args.monocular,
            "fps": args.fps,
            "video_id": None,
            "serial_number": None,
            "physical_path": None,
        },
        "left_wrist_camera": {
            "enable_zmq": False,
            "zmq_port": 55556,
            "enable_webrtc": False,
            "webrtc_port": 60002,
            "webrtc_codec": args.codec,
            "type": "testpattern",
            "image_shape": [args.height, args.width // 2],
            "binocular": False,
            "fps": args.fps,
            "video_id": None,
            "serial_number": None,
            "physical_path": None,
        },
        "right_wrist_camera": {
            "enable_zmq": False,
            "zmq_port": 55557,
            "enable_webrtc": False,
            "webrtc_port": 60003,
            "webrtc_codec": args.codec,
            "type": "testpattern",
            "image_shape": [args.height, args.width // 2],
            "binocular": False,
            "fps": args.fps,
            "video_id": None,
            "serial_number": None,
            "physical_path": None,
        },
    }


def make_frame(width: int, height: int, frame_index: int, binocular: bool) -> np.ndarray:
    t = frame_index / 30.0
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xv = np.tile(x, (height, 1))
    yv = np.tile(y[:, None], (1, width))
    wave = ((np.sin((np.arange(width)[None, :] / 42.0) + t) + 1.0) * 60.0).astype(np.uint8)

    frame = np.dstack(
        (
            (xv // 2 + wave) % 255,
            (yv + frame_index * 2) % 255,
            ((255 - xv) // 2 + 80) % 255,
        )
    ).astype(np.uint8)

    cv2.rectangle(frame, (18, 18), (width - 18, height - 18), (255, 255, 255), 2)
    if binocular:
        mid = width // 2
        cv2.line(frame, (mid, 0), (mid, height), (255, 255, 255), 2)
        cv2.putText(frame, "LEFT TEST VIEW", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(frame, "LEFT TEST VIEW", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT TEST VIEW", (mid + 40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(frame, "RIGHT TEST VIEW", (mid + 40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "XR TELEOP TEST VIEW", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 4, cv2.LINE_AA)
        cv2.putText(frame, "XR TELEOP TEST VIEW", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    label = f"frame {frame_index:06d}  {time.strftime('%H:%M:%S')}"
    cv2.putText(frame, label, (40, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 4, cv2.LINE_AA)
    cv2.putText(frame, label, (40, height - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a teleimager-compatible WebRTC test-pattern server.")
    parser.add_argument("--request-port", type=int, default=60000, help="ZMQ camera-config request port.")
    parser.add_argument("--webrtc-port", type=int, default=60001, help="Head-camera WebRTC offer port.")
    parser.add_argument("--width", type=int, default=1280, help="Generated frame width.")
    parser.add_argument("--height", type=int, default=480, help="Generated frame height.")
    parser.add_argument("--fps", type=float, default=30.0, help="Generated frame rate.")
    parser.add_argument("--codec", choices=["h264", "vp8"], default="h264", help="Preferred WebRTC codec.")
    parser.add_argument("--monocular", action="store_true", help="Advertise the generated image as monocular.")
    return parser.parse_args()


def main() -> int:
    logging_mp.basicConfig(level=logging_mp.INFO)
    args = parse_args()
    stop_event = threading.Event()
    config = build_config(args)

    def stop(_signum=None, _frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    responder = ZMQ_Responser(config, port=args.request_port)
    webrtc_manager = WebRTC_PublisherManager.get_instance()
    interval = 1.0 / args.fps
    frame_index = 0

    logger.info(
        "[Test Image Server] Config on tcp://0.0.0.0:%s, WebRTC offer on https://0.0.0.0:%s/offer",
        args.request_port,
        args.webrtc_port,
    )
    logger.info("[Test Image Server] Press Ctrl+C to exit.")
    print(
        f"Test image server running: config tcp://0.0.0.0:{args.request_port}, "
        f"WebRTC https://0.0.0.0:{args.webrtc_port}/offer",
        flush=True,
    )
    print("Press Ctrl+C to exit.", flush=True)

    try:
        next_frame_time = time.monotonic()
        while not stop_event.is_set():
            frame = make_frame(args.width, args.height, frame_index, not args.monocular)
            webrtc_manager.publish(frame, args.webrtc_port, codec_pref=args.codec)
            frame_index += 1
            next_frame_time += interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                stop_event.wait(sleep_time)
            else:
                next_frame_time = time.monotonic()
    finally:
        responder.stop()
        webrtc_manager.close()
        logger.info("[Test Image Server] Stopped.")
        print("Test image server stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
