#!/usr/bin/env python3
"""TeleImager-compatible bridge for the built-in Unitree videohub camera."""

from __future__ import annotations

import argparse
from pathlib import Path
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
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.video.video_client import VideoClient


logger = logging_mp.getLogger(__name__)


def even_dimension(value: float) -> int:
    size = max(2, int(round(value)))
    return size if size % 2 == 0 else size - 1


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_width) / float(width))
    out_width = even_dimension(width * scale)
    out_height = even_dimension(height * scale)
    if out_width == width and out_height == height:
        return frame
    return cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA)


def decode_videohub_frame(data) -> np.ndarray | None:
    if not data:
        return None
    encoded = np.frombuffer(bytes(data), dtype=np.uint8)
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def get_videohub_frame(client: VideoClient, max_width: int) -> tuple[int, int | None, np.ndarray | None]:
    code, data = client.GetImageSample()
    data_size = len(data) if data is not None else None
    if code != 0 or not data:
        return code, data_size, None
    frame = decode_videohub_frame(data)
    if frame is None:
        return code, data_size, None
    return code, data_size, resize_frame(frame, max_width)


def wait_for_first_frame(
    client: VideoClient,
    max_width: int,
    timeout_s: float,
) -> tuple[int, int, np.ndarray]:
    deadline = time.monotonic() + timeout_s
    last_code = None
    last_size = None
    while time.monotonic() < deadline:
        code, data_size, frame = get_videohub_frame(client, max_width)
        last_code = code
        last_size = data_size
        if frame is not None:
            return code, int(data_size or 0), frame
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for videohub frame; last code={last_code}, bytes={last_size}")


def build_config(args: argparse.Namespace, frame: np.ndarray) -> dict:
    height, width = frame.shape[:2]
    disabled_wrist = {
        "enable_zmq": False,
        "zmq_port": 0,
        "enable_webrtc": False,
        "webrtc_port": 0,
        "webrtc_codec": args.codec,
        "type": "disabled",
        "image_shape": [height, width],
        "binocular": False,
        "fps": args.fps,
        "video_id": None,
        "serial_number": None,
        "physical_path": None,
    }
    return {
        "head_camera": {
            "enable_zmq": False,
            "zmq_port": 55555,
            "enable_webrtc": True,
            "webrtc_port": args.webrtc_port,
            "webrtc_codec": args.codec,
            "type": "unitree_videohub",
            "image_shape": [height, width],
            "binocular": False,
            "fps": args.fps,
            "video_id": None,
            "serial_number": None,
            "physical_path": None,
        },
        "left_wrist_camera": dict(disabled_wrist),
        "right_wrist_camera": dict(disabled_wrist),
    }


def save_preview(path: str, frame: np.ndarray) -> None:
    preview_path = Path(path).expanduser()
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(preview_path), frame):
        raise RuntimeError(f"failed to write preview image to {preview_path}")
    print(f"Saved preview image: {preview_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TeleImager-compatible bridge for the Unitree videohub camera.")
    parser.add_argument("--interface", default="eno1", help="DDS network interface for Unitree videohub.")
    parser.add_argument("--request-port", type=int, default=60000, help="ZMQ camera-config request port.")
    parser.add_argument("--webrtc-port", type=int, default=60001, help="Head-camera WebRTC offer port.")
    parser.add_argument("--fps", type=float, default=20.0, help="Target WebRTC publish rate.")
    parser.add_argument("--max-width", type=int, default=1280, help="Maximum output frame width.")
    parser.add_argument("--codec", choices=["h264", "vp8"], default="h264", help="Preferred WebRTC codec.")
    parser.add_argument("--rpc-timeout", type=float, default=2.0, help="Seconds for each videohub RPC call.")
    parser.add_argument("--startup-timeout", type=float, default=10.0, help="Seconds to wait for the first valid frame.")
    parser.add_argument("--save-preview", default=None, help="Optional path to write the first decoded/resized frame.")
    parser.add_argument("--preview-only", action="store_true", help="Save/check one frame and exit without starting WebRTC.")
    args = parser.parse_args()
    if args.fps <= 0.0:
        parser.error("--fps must be positive.")
    if args.max_width <= 0:
        parser.error("--max-width must be positive.")
    if args.rpc_timeout <= 0.0:
        parser.error("--rpc-timeout must be positive.")
    if args.startup_timeout <= 0.0:
        parser.error("--startup-timeout must be positive.")
    return args


def main() -> int:
    logging_mp.basicConfig(level=logging_mp.INFO)
    args = parse_args()
    stop_event = threading.Event()

    def stop(_signum=None, _frame=None) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ChannelFactoryInitialize(0, networkInterface=args.interface)
    client = VideoClient()
    client.SetTimeout(args.rpc_timeout)
    client.Init()

    code, data_size, first_frame = wait_for_first_frame(client, args.max_width, args.startup_timeout)
    height, width = first_frame.shape[:2]
    print(f"videohub code={code}, bytes={data_size}, output_shape=({height}, {width}, 3)", flush=True)
    if args.save_preview:
        save_preview(args.save_preview, first_frame)
    if args.preview_only:
        return 0

    config = build_config(args, first_frame)
    responder = ZMQ_Responser(config, port=args.request_port)
    webrtc_manager = WebRTC_PublisherManager.get_instance()
    interval = 1.0 / args.fps
    frame_count = 0
    last_warn_time = 0.0

    logger.info(
        "[Videohub Image Server] Config on tcp://0.0.0.0:%s, WebRTC offer on https://0.0.0.0:%s/offer",
        args.request_port,
        args.webrtc_port,
    )
    print(
        f"Videohub image server running: config tcp://0.0.0.0:{args.request_port}, "
        f"WebRTC https://0.0.0.0:{args.webrtc_port}/offer",
        flush=True,
    )
    print("Press Ctrl+C to exit.", flush=True)

    try:
        next_frame_time = time.monotonic()
        current_frame = first_frame
        while not stop_event.is_set():
            code, data_size, frame = get_videohub_frame(client, args.max_width)
            if frame is not None:
                current_frame = frame
            else:
                now = time.monotonic()
                if now - last_warn_time > 2.0:
                    logger.warning("[Videohub Image Server] No valid frame: code=%s, bytes=%s; reusing last frame.", code, data_size)
                    last_warn_time = now

            webrtc_manager.publish(current_frame, args.webrtc_port, codec_pref=args.codec)
            frame_count += 1
            if frame_count % max(1, int(args.fps * 5.0)) == 0:
                logger.info("[Videohub Image Server] Published %s frames.", frame_count)

            next_frame_time += interval
            sleep_time = next_frame_time - time.monotonic()
            if sleep_time > 0:
                stop_event.wait(sleep_time)
            else:
                next_frame_time = time.monotonic()
    finally:
        responder.stop()
        webrtc_manager.close()
        logger.info("[Videohub Image Server] Stopped.")
        print("Videohub image server stopped.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
