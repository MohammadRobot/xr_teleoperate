#!/usr/bin/env python3
"""Safe preflight and guarded launcher for G1 + Quest 3 controller teleop.

The default mode performs read-only checks: host network, image server config,
Python imports, Vuer certificate files, and G1 lowstate DDS subscription.
It does not create arm or hand command publishers. Use --launch together with
--operator-ready only after the robot area is clear and a human operator is
ready to stop/damp the robot.
"""

from __future__ import annotations

import argparse
import importlib
import ipaddress
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


TELEOP_DIR = Path(__file__).resolve().parent
REPO_ROOT = TELEOP_DIR.parent


def remove_user_site() -> None:
    """Keep ~/.local packages from shadowing the conda env packages."""
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    try:
        import site

        user_site = Path(site.getusersitepackages()).resolve()
        sys.path[:] = [
            path
            for path in sys.path
            if not path or Path(path).resolve() != user_site
        ]
    except Exception:
        pass


def print_check(name: str, ok: bool, detail: str) -> bool:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    return ok


def run_command(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )


def check_python_imports(no_camera: bool = False) -> bool:
    remove_user_site()
    sys.path.insert(0, str(REPO_ROOT))
    modules = [
        "numpy",
        "pinocchio",
        "teleop.utils.logging_compat",
        "televuer",
        "unitree_sdk2py",
        "sshkeyboard",
        "meshcat",
        "logging_mp",
    ]
    if not no_camera:
        modules.extend([
            "teleimager",
            "teleimager.image_client",
            "teleimager.image_server",
        ])
    failures: list[str] = []
    versions: list[str] = []
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "")
            versions.append(f"{module_name}{' ' + version if version else ''}")
        except Exception as exc:
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")

    if failures:
        return print_check("Python imports", False, "; ".join(failures))
    return print_check("Python imports", True, ", ".join(versions))


def get_interface_addresses(interface: str) -> list[ipaddress.IPv4Interface]:
    result = run_command(["ip", "-4", "-o", "addr", "show", "dev", interface], timeout=2)
    if result.returncode != 0:
        return []

    addresses: list[ipaddress.IPv4Interface] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if "inet" not in parts:
            continue
        inet_index = parts.index("inet")
        if inet_index + 1 >= len(parts):
            continue
        try:
            addresses.append(ipaddress.IPv4Interface(parts[inet_index + 1]))
        except ValueError:
            continue
    return addresses


def check_network(interface: str, expected_network: str, allow_non_unitree: bool) -> tuple[bool, str | None]:
    addresses = get_interface_addresses(interface)
    if not addresses:
        print_check("Network interface", False, f"{interface} has no IPv4 address")
        return False, None

    network = ipaddress.IPv4Network(expected_network, strict=False)
    selected = addresses[0]
    host_ip = str(selected.ip)
    if selected.ip in network:
        print_check("Network interface", True, f"{interface} is {selected} in {network}")
        return True, host_ip

    detail = f"{interface} is {selected}, expected {network}"
    if allow_non_unitree:
        print_check("Network interface", True, detail + " (allowed by flag)")
        return True, host_ip
    print_check("Network interface", False, detail)
    return False, host_ip


def check_ping(host: str, skip: bool, label: str = "Image server ping") -> bool:
    if skip:
        return print_check(label, True, "skipped")
    result = run_command(["ping", "-c", "1", "-W", "1", host], timeout=2)
    if result.returncode == 0:
        first_line = result.stdout.splitlines()[1] if len(result.stdout.splitlines()) > 1 else "reachable"
        return print_check(label, True, first_line.strip())
    return print_check(label, False, f"{host} is not reachable")


def request_camera_config(host: str, port: int, timeout_ms: int, skip: bool) -> tuple[bool, dict[str, Any] | None]:
    if skip:
        print_check("Teleimager config", True, "skipped")
        return True, None

    try:
        import zmq
    except Exception as exc:
        print_check("Teleimager config", False, f"pyzmq import failed: {exc}")
        return False, None

    context = zmq.Context()
    socket_req = context.socket(zmq.REQ)
    socket_req.setsockopt(zmq.LINGER, 0)
    poller = zmq.Poller()
    poller.register(socket_req, zmq.POLLIN)

    try:
        socket_req.connect(f"tcp://{host}:{port}")
        socket_req.send(b"GET_DATA")
        events = dict(poller.poll(timeout=timeout_ms))
        if socket_req not in events:
            print_check("Teleimager config", False, f"no response from {host}:{port}")
            return False, None

        config = socket_req.recv_json()
        head = config.get("head_camera", {})
        if not head:
            print_check("Teleimager config", False, "missing head_camera config")
            return False, config

        transport_ok = bool(head.get("enable_webrtc") or head.get("enable_zmq"))
        if not transport_ok:
            print_check("Teleimager config", False, "head camera has neither WebRTC nor ZMQ enabled")
            return False, config

        if not bool(head.get("enable_webrtc")):
            print_check("Teleimager config", False, "head camera WebRTC is disabled")
            return False, config

        webrtc_port = head.get("webrtc_port", "unknown")
        image_shape = head.get("image_shape", "unknown")
        print_check("Teleimager config", True, f"head WebRTC on https://{host}:{webrtc_port}, shape={image_shape}")
        return True, config
    except Exception as exc:
        print_check("Teleimager config", False, f"{type(exc).__name__}: {exc}")
        return False, None
    finally:
        socket_req.close()
        context.term()


def check_vuer_certs() -> bool:
    env_cert = os.getenv("XR_TELEOP_CERT")
    env_key = os.getenv("XR_TELEOP_KEY")
    candidates: list[tuple[Path, Path, str]] = []
    if env_cert and env_key:
        candidates.append((Path(env_cert), Path(env_key), "environment"))
    candidates.append((Path.home() / ".config/xr_teleoperate/cert.pem", Path.home() / ".config/xr_teleoperate/key.pem", "user config"))
    candidates.append((TELEOP_DIR / "televuer/cert.pem", TELEOP_DIR / "televuer/key.pem", "televuer directory"))

    for cert, key, source in candidates:
        if cert.exists() and key.exists():
            return print_check("Vuer certificates", True, f"{source}: {cert}, {key}")
    return print_check("Vuer certificates", False, "cert.pem/key.pem not found")


def check_dds_lowstate(interface: str, timeout_s: float, skip: bool) -> bool:
    if skip:
        return print_check("DDS lowstate", True, "skipped")

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as hg_LowState
    except Exception as exc:
        return print_check("DDS lowstate", False, f"import failed: {exc}")

    received = {"ok": False, "mode_machine": None}

    def callback(msg: Any) -> None:
        if msg is None:
            return
        received["ok"] = True
        received["mode_machine"] = getattr(msg, "mode_machine", None)

    try:
        ChannelFactoryInitialize(0, networkInterface=interface)
        subscriber = ChannelSubscriber("rt/lowstate", hg_LowState)
        try:
            subscriber.Init(callback, 10)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and not received["ok"]:
                time.sleep(0.05)
        except TypeError:
            subscriber.Init()
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and not received["ok"]:
                msg = subscriber.Read()
                if msg is not None:
                    callback(msg)
                    break
                time.sleep(0.05)

        if received["ok"]:
            return print_check("DDS lowstate", True, f"received rt/lowstate, mode_machine={received['mode_machine']}")
        return print_check("DDS lowstate", False, f"no rt/lowstate within {timeout_s:.1f}s on {interface}")
    except Exception as exc:
        return print_check("DDS lowstate", False, f"{type(exc).__name__}: {exc}")


def check_dex3_state(interface: str, timeout_s: float, skip_hands: bool, skip_dds: bool) -> bool:
    if skip_hands:
        return print_check("Dex3 DDS state", True, "skipped (--skip-hands)")
    if skip_dds:
        return print_check("Dex3 DDS state", True, "skipped (--skip-dds)")

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_
    except Exception as exc:
        return print_check("Dex3 DDS state", False, f"import failed: {exc}")

    received = {"left": False, "right": False}

    def left_callback(msg: Any) -> None:
        if msg is not None:
            received["left"] = True

    def right_callback(msg: Any) -> None:
        if msg is not None:
            received["right"] = True

    try:
        ChannelFactoryInitialize(0, networkInterface=interface)
        left_subscriber = ChannelSubscriber("rt/dex3/left/state", HandState_)
        right_subscriber = ChannelSubscriber("rt/dex3/right/state", HandState_)
        try:
            left_subscriber.Init(left_callback, 10)
            right_subscriber.Init(right_callback, 10)
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and not (received["left"] and received["right"]):
                time.sleep(0.05)
        except TypeError:
            left_subscriber.Init()
            right_subscriber.Init()
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline and not (received["left"] and received["right"]):
                if not received["left"] and left_subscriber.Read() is not None:
                    received["left"] = True
                if not received["right"] and right_subscriber.Read() is not None:
                    received["right"] = True
                time.sleep(0.05)

        if received["left"] and received["right"]:
            return print_check("Dex3 DDS state", True, "received rt/dex3/left/state and rt/dex3/right/state")

        missing = [name for name, ok in received.items() if not ok]
        return print_check("Dex3 DDS state", False, f"missing {', '.join(missing)} state within {timeout_s:.1f}s")
    except Exception as exc:
        return print_check("Dex3 DDS state", False, f"{type(exc).__name__}: {exc}")


def build_teleop_command(args: argparse.Namespace) -> list[str]:
    end_effector = "none" if args.skip_hands else "dex3"
    command = [
        sys.executable,
        "teleop_hand_and_arm.py",
        "--arm=G1_29",
        f"--ee={end_effector}",
        "--input-mode=controller",
        "--display-mode=pass-through" if args.no_camera else "--display-mode=immersive",
        f"--network-interface={args.interface}",
        "--motion",
        f"--dex3-kp={args.dex3_kp}",
        f"--dex3-kp-boost={args.dex3_kp_boost}",
        f"--dex3-kd={args.dex3_kd}",
    ]
    if args.no_camera:
        command.append("--no-camera")
    else:
        command.append(f"--img-server-ip={args.img_server_ip}")
    if args.frequency is not None:
        command.append(f"--frequency={args.frequency}")
    if args.record:
        command.append("--record")
    if args.headless:
        command.append("--headless")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G1_29 + Dex3 + Quest 3 controller dry-run preflight and launcher.")
    parser.add_argument("--interface", default="eno1", help="DDS/network interface to use.")
    parser.add_argument("--expected-network", default="192.168.123.0/24", help="Expected Unitree network CIDR.")
    parser.add_argument("--img-server-ip", default=None, help="Teleimager IP address. Defaults to PC2 192.168.123.164 unless --local-image-server is set.")
    parser.add_argument("--local-image-server", action="store_true", help="Use this host's interface IP as the teleimager server.")
    parser.add_argument("--no-camera", action="store_true", help="Skip Teleimager/WebRTC checks and launch teleop in pass-through controller mode.")
    parser.add_argument("--skip-hands", action="store_true", help="Launch with --ee=none so arm/walking tests do not wait for Dex3 hand DDS state.")
    parser.add_argument("--img-request-port", type=int, default=60000, help="Teleimager config request port.")
    parser.add_argument("--img-timeout-ms", type=int, default=1000, help="Teleimager config timeout in milliseconds.")
    parser.add_argument("--dds-timeout", type=float, default=5.0, help="Seconds to wait for rt/lowstate.")
    parser.add_argument("--allow-non-unitree-network", action="store_true", help="Warn but allow non-192.168.123.x host IP.")
    parser.add_argument(
        "--skip-image-server-ping",
        "--skip-pc2-ping",
        dest="skip_image_server_ping",
        action="store_true",
        help="Skip pinging the selected image server.",
    )
    parser.add_argument("--skip-image-config", action="store_true", help="Skip the teleimager config request.")
    parser.add_argument("--skip-dds", action="store_true", help="Skip DDS lowstate subscription check.")
    parser.add_argument("--launch", action="store_true", help="Launch teleop after all checks pass.")
    parser.add_argument("--operator-ready", action="store_true", help="Required with --launch to acknowledge robot motion risk.")
    parser.add_argument("--frequency", type=float, default=30.0, help="Teleop frequency passed to teleop_hand_and_arm.py.")
    parser.add_argument("--dex3-kp", type=float, default=0.8, help="Dex3 normal controller-mode position stiffness passed to teleop.")
    parser.add_argument("--dex3-kp-boost", type=float, default=1.2, help="Dex3 boosted controller-mode position stiffness passed to teleop.")
    parser.add_argument("--dex3-kd", type=float, default=0.2, help="Dex3 controller-mode damping passed to teleop.")
    parser.add_argument("--record", action="store_true", help="Pass --record to teleop.")
    parser.add_argument("--headless", action="store_true", help="Pass --headless to teleop.")
    args = parser.parse_args()
    if args.no_camera and args.local_image_server:
        parser.error("--no-camera cannot be combined with --local-image-server.")
    if args.no_camera and args.record:
        parser.error("--no-camera cannot be combined with --record.")
    if args.dex3_kp < 0.0 or args.dex3_kp_boost < 0.0 or args.dex3_kd < 0.0:
        parser.error("--dex3-kp, --dex3-kp-boost, and --dex3-kd must be non-negative.")
    return args


def main() -> int:
    args = parse_args()
    print("G1_29 + Dex3 + Quest 3 controller dry-run preflight")
    print(f"Repo: {REPO_ROOT}")

    ok = True
    ok = check_python_imports(no_camera=args.no_camera) and ok
    network_ok, host_ip = check_network(args.interface, args.expected_network, args.allow_non_unitree_network)
    ok = network_ok and ok
    config = None
    if args.no_camera:
        print_check("Image server selection", True, "no-camera mode; skipping Teleimager/WebRTC")
    elif args.local_image_server:
        if host_ip is None:
            print_check("Image server selection", False, f"cannot use local image server without an IP on {args.interface}")
            ok = False
        else:
            args.img_server_ip = host_ip
            print_check("Image server selection", True, f"using local host image server at {args.img_server_ip}")
            print("Start local image server in another terminal with:")
            print(f"  cd {TELEOP_DIR}")
            print("  export PYTHONNOUSERSITE=1")
            print("  python run_local_image_server.py")
            print("If this PC has no local camera attached, use:")
            print("  python run_test_image_server.py")
    elif args.img_server_ip is None:
        args.img_server_ip = "192.168.123.164"
        print_check("Image server selection", True, f"using PC2 image server at {args.img_server_ip}")
    else:
        print_check("Image server selection", True, f"using image server at {args.img_server_ip}")
    ok = check_vuer_certs() and ok
    if args.no_camera:
        print_check("Teleimager config", True, "skipped (--no-camera)")
    else:
        ping_label = "Local image server ping" if args.local_image_server else "PC2 image server ping"
        ok = check_ping(args.img_server_ip, args.skip_image_server_ping, ping_label) and ok
        image_ok, config = request_camera_config(args.img_server_ip, args.img_request_port, args.img_timeout_ms, args.skip_image_config)
        ok = image_ok and ok
    ok = check_dds_lowstate(args.interface, args.dds_timeout, args.skip_dds) and ok
    ok = check_dex3_state(args.interface, args.dds_timeout, args.skip_hands, args.skip_dds) and ok

    if host_ip:
        quest_url = f"https://{host_ip}:8012/?ws=wss://{host_ip}:8012"
        print(f"Quest 3 URL: {quest_url}")
    if config:
        head = config.get("head_camera", {})
        print(f"Head WebRTC offer: https://{args.img_server_ip}:{head.get('webrtc_port', 60001)}/offer")

    command = build_teleop_command(args)
    print(f"Teleop working directory: {TELEOP_DIR}")
    print("Teleop command:")
    print("  " + " ".join(command))

    if not ok:
        print("Preflight failed. Fix the failed checks before launching teleop.")
        return 1

    if not args.launch:
        print("Preflight passed. Re-run with --launch --operator-ready to start teleop.")
        return 0

    if not args.operator_ready:
        print("Refusing to launch without --operator-ready.")
        return 2

    print("Launching teleop. Keep the robot clear. Press r or left Y/B to sync arms, left Y/B to pause/resume arms, and q/right A to exit.")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    os.chdir(TELEOP_DIR)
    os.execvpe(command[0], command, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
