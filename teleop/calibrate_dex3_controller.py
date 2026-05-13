#!/usr/bin/env python3
"""Teach Dex3 controller raw-radian calibration endpoints from current DDS hand state."""

from __future__ import annotations

import argparse
import math
import select
import shutil
import sys
import termios
import time
import tty
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
ASSET_DIR = REPO_ROOT / "assets" / "unitree_hand"
DEFAULT_CALIBRATION_PATH = ASSET_DIR / "dex3_controller_calibration.yml"

DEX3_LEFT_STATE_TOPIC = "rt/dex3/left/state"
DEX3_RIGHT_STATE_TOPIC = "rt/dex3/right/state"
DEX3_LEFT_COMMAND_TOPIC = "rt/dex3/left/cmd"
DEX3_RIGHT_COMMAND_TOPIC = "rt/dex3/right/cmd"

ENDPOINTS = ("open", "grip_close", "trigger_close")
ENDPOINT_RAW_KEYS = {
    "open": "open_rad",
    "grip_close": "grip_close_rad",
    "trigger_close": "trigger_close_rad",
}
ENDPOINT_FRACTION_DEFAULTS = {"open": 0.0, "grip_close": 1.0, "trigger_close": 1.0}

JOINT_SPECS = {
    "left": [
        ("thumb_0", "left_hand_thumb_0_joint", 0),
        ("thumb_1", "left_hand_thumb_1_joint", 1),
        ("thumb_2", "left_hand_thumb_2_joint", 2),
        ("middle_0", "left_hand_middle_0_joint", 3),
        ("middle_1", "left_hand_middle_1_joint", 4),
        ("index_0", "left_hand_index_0_joint", 5),
        ("index_1", "left_hand_index_1_joint", 6),
    ],
    "right": [
        ("thumb_0", "right_hand_thumb_0_joint", 0),
        ("thumb_1", "right_hand_thumb_1_joint", 1),
        ("thumb_2", "right_hand_thumb_2_joint", 2),
        ("middle_0", "right_hand_middle_0_joint", 3),
        ("middle_1", "right_hand_middle_1_joint", 4),
        ("index_0", "right_hand_index_0_joint", 5),
        ("index_1", "right_hand_index_1_joint", 6),
    ],
}


def clamp_fraction(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def fraction_to_angle(fraction: float, joint_range: dict[str, float]) -> float:
    fraction = clamp_fraction(fraction)
    return joint_range["open_angle"] + fraction * (joint_range["close_angle"] - joint_range["open_angle"])


def raw_endpoint_from_config(endpoint_config: dict[str, Any], endpoint: str, joint_range: dict[str, float]) -> float:
    raw_key = ENDPOINT_RAW_KEYS[endpoint]
    if raw_key in endpoint_config:
        return float(endpoint_config[raw_key])
    if endpoint == "open":
        fraction = endpoint_config.get("open", ENDPOINT_FRACTION_DEFAULTS["open"])
    elif endpoint == "grip_close":
        fraction = endpoint_config.get("grip_close", endpoint_config.get("close", ENDPOINT_FRACTION_DEFAULTS["grip_close"]))
    else:
        fraction = endpoint_config.get("trigger_close", endpoint_config.get("close", ENDPOINT_FRACTION_DEFAULTS["trigger_close"]))
    return fraction_to_angle(float(fraction), joint_range)


def endpoint_from_config(endpoint_config: Any, joint_range: dict[str, float]) -> dict[str, Any]:
    if endpoint_config is None:
        endpoint_config = {}
    if not isinstance(endpoint_config, dict):
        raise ValueError("calibration endpoint must be a mapping")
    return {
        "open_rad": raw_endpoint_from_config(endpoint_config, "open", joint_range),
        "grip_close_rad": raw_endpoint_from_config(endpoint_config, "grip_close", joint_range),
        "trigger_close_rad": raw_endpoint_from_config(endpoint_config, "trigger_close", joint_range),
    }


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} top-level YAML value must be a mapping")
    return data


def write_yaml_with_backup(path: Path, config: dict[str, Any]) -> Path | None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if path.exists():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = path.with_name(f"{path.name}.{timestamp}.bak")
        shutil.copy2(path, backup_path)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=None)
    return backup_path


def read_urdf_limits(hand: str) -> dict[str, tuple[float, float]]:
    urdf_path = ASSET_DIR / f"unitree_dex3_{hand}.urdf"
    limits: dict[str, tuple[float, float]] = {}
    root = ET.parse(urdf_path).getroot()
    for joint in root.findall("joint"):
        name = joint.get("name")
        limit = joint.find("limit")
        if name is None or limit is None:
            continue
        limits[name] = (float(limit.get("lower", "0")), float(limit.get("upper", "0")))
    return limits


def default_joint_ranges(hand: str) -> dict[str, dict[str, float]]:
    limits = read_urdf_limits(hand)
    ranges: dict[str, dict[str, float]] = {}
    for joint_key, urdf_name, _ in JOINT_SPECS[hand]:
        lower, upper = limits.get(urdf_name, (0.0, 0.0))
        open_angle = clamp_angle(0.0, lower, upper)
        close_angle = upper if upper > 0.0 else lower
        close_angle = clamp_angle(close_angle, lower, upper)
        ranges[joint_key] = {
            "lower": lower,
            "upper": upper,
            "open_angle": open_angle,
            "close_angle": close_angle,
        }
    return ranges


def clamp_angle(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def ensure_hand_joint_config(config: dict[str, Any], hand: str) -> dict[str, dict[str, Any]]:
    hand_config = config.setdefault(hand, {})
    if not isinstance(hand_config, dict):
        raise ValueError(f"{hand} must be a mapping")

    joint_config = hand_config.get("joints")
    if joint_config is not None and not isinstance(joint_config, dict):
        raise ValueError(f"{hand}.joints must be a mapping")

    ranges = default_joint_ranges(hand)
    raw_joints: dict[str, dict[str, Any]] = {}
    for joint_key, _, _ in JOINT_SPECS[hand]:
        if isinstance(joint_config, dict) and joint_key in joint_config:
            source = joint_config[joint_key]
        else:
            finger = joint_key.split("_", 1)[0]
            source = hand_config.get(finger, {})
        raw_joints[joint_key] = endpoint_from_config(source, ranges[joint_key])
    hand_config["joints"] = raw_joints
    return raw_joints


def update_config_endpoint(
    config: dict[str, Any],
    captured: dict[str, dict[str, float]],
    endpoint: str,
) -> dict[str, Any]:
    if endpoint not in ENDPOINTS:
        raise ValueError(f"endpoint must be one of {', '.join(ENDPOINTS)}")
    raw_key = ENDPOINT_RAW_KEYS[endpoint]
    for hand, joint_values in captured.items():
        joint_config = ensure_hand_joint_config(config, hand)
        for joint_key, value in joint_values.items():
            joint_config[joint_key][raw_key] = float(value)
    return config


def extract_hand_state(msg: Any, hand: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for joint_key, _, dds_index in JOINT_SPECS[hand]:
        values[joint_key] = float(msg.motor_state[dds_index].q)
    return values


def collect_dds_samples(
    interface: str,
    hands: list[str],
    sample_count: int,
    timeout_s: float,
    sample_period_s: float,
) -> dict[str, list[dict[str, float]]]:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandState_

    ChannelFactoryInitialize(0, networkInterface=interface)
    subscribers = {}
    for hand in hands:
        topic = DEX3_LEFT_STATE_TOPIC if hand == "left" else DEX3_RIGHT_STATE_TOPIC
        subscriber = ChannelSubscriber(topic, HandState_)
        subscriber.Init()
        subscribers[hand] = subscriber

    samples: dict[str, list[dict[str, float]]] = {hand: [] for hand in hands}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for hand, subscriber in subscribers.items():
            if len(samples[hand]) >= sample_count:
                continue
            msg = subscriber.Read()
            if msg is not None:
                samples[hand].append(extract_hand_state(msg, hand))
        if all(len(hand_samples) >= sample_count for hand_samples in samples.values()):
            return samples
        time.sleep(sample_period_s)

    missing = [f"{hand}: {len(values)}/{sample_count}" for hand, values in samples.items()]
    raise TimeoutError(f"timed out waiting for Dex3 state samples ({', '.join(missing)})")


def motor_mode_uint8(motor_id: int, status: int = 0x01, timeout: int = 0) -> int:
    return (motor_id & 0x0F) | ((status & 0x07) << 4) | ((timeout & 0x01) << 7)


def command_topic(hand: str) -> str:
    return DEX3_LEFT_COMMAND_TOPIC if hand == "left" else DEX3_RIGHT_COMMAND_TOPIC


def state_topic(hand: str) -> str:
    return DEX3_LEFT_STATE_TOPIC if hand == "left" else DEX3_RIGHT_STATE_TOPIC


def update_hand_command_msg(msg: Any, hand: str, target: dict[str, float], kp: float, kd: float) -> None:
    for joint_key, _, dds_index in JOINT_SPECS[hand]:
        motor_cmd = msg.motor_cmd[dds_index]
        motor_cmd.mode = motor_mode_uint8(dds_index)
        motor_cmd.q = target[joint_key]
        motor_cmd.dq = 0.0
        motor_cmd.tau = 0.0
        motor_cmd.kp = kp
        motor_cmd.kd = kd


def read_hand_state_sample(subscriber: Any, hand: str, timeout_s: float) -> dict[str, float]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        msg = subscriber.Read()
        if msg is not None:
            return extract_hand_state(msg, hand)
        time.sleep(0.01)
    raise TimeoutError(f"timed out waiting for {state_topic(hand)}")


def average_jog_state_samples(
    subscriber: Any,
    publisher: Any,
    msg: Any,
    hand: str,
    target: dict[str, float],
    sample_count: int,
    timeout_s: float,
    sample_period_s: float,
    kp: float,
    kd: float,
) -> dict[str, list[dict[str, float]]]:
    samples = {hand: []}
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and len(samples[hand]) < sample_count:
        update_hand_command_msg(msg, hand, target, kp, kd)
        publisher.Write(msg)
        state_msg = subscriber.Read()
        if state_msg is not None:
            samples[hand].append(extract_hand_state(state_msg, hand))
        time.sleep(sample_period_s)
    if len(samples[hand]) < sample_count:
        raise TimeoutError(f"timed out waiting for jog save samples ({len(samples[hand])}/{sample_count})")
    return samples


class RawTerminal:
    def __init__(self) -> None:
        self._settings = None

    def __enter__(self) -> "RawTerminal":
        if sys.stdin.isatty():
            self._settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._settings)


def read_key() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0.0)
    if not readable:
        return None
    return sys.stdin.read(1)


def joint_key_by_number(hand: str, key: str) -> str | None:
    if key not in {"1", "2", "3", "4", "5", "6", "7"}:
        return None
    return JOINT_SPECS[hand][int(key) - 1][0]


def print_jog_controls(hand: str, endpoint: str, selected_joint: str, target: dict[str, float]) -> None:
    print(f"\nJogging {hand} Dex3 for endpoint '{endpoint}'.")
    print("Controls:")
    print("  1-7 select joint")
    print("  a/d or left/right bracket = decrease/increase selected joint")
    print("  z/x = step selected joint toward open/close")
    print("  o = selected joint to URDF open, c = selected joint to URDF close")
    print("  p = print targets, s = save current measured pose, q = quit")
    print("\nJoint order:")
    for idx, (joint_key, _, _) in enumerate(JOINT_SPECS[hand], start=1):
        marker = "*" if joint_key == selected_joint else " "
        print(f"  {idx}. {marker}{joint_key}: {target[joint_key]:.4f} rad")


def print_jog_targets(hand: str, selected_joint: str, target: dict[str, float]) -> None:
    print("\nCurrent jog targets:")
    for joint_key, _, _ in JOINT_SPECS[hand]:
        marker = "*" if joint_key == selected_joint else " "
        print(f"  {marker} {joint_key:<8} {target[joint_key]: .4f} rad")


def step_toward(current: float, destination: float, step: float) -> float:
    if abs(destination - current) <= step:
        return destination
    if destination > current:
        return current + step
    return current - step


def save_captured_endpoint(
    args: argparse.Namespace,
    means: dict[str, dict[str, float]],
    stddevs: dict[str, dict[str, float]],
) -> int:
    captured, range_warnings = captured_raw_angles(means)
    stability_warnings = print_capture(args.endpoint, means, stddevs, captured, args.max_std)

    warnings = range_warnings + stability_warnings
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    config = load_yaml(args.calibration_path)
    update_config_endpoint(config, captured, args.endpoint)

    if args.dry_run:
        print(f"\nDry run only. {args.calibration_path} was not changed.")
        return 0

    if not args.yes:
        answer = input(f"\nWrite {args.endpoint} values to {args.calibration_path}? [y/N] ").strip().lower()
        if answer not in {"y", "yes"}:
            print("Calibration not written.")
            return 0

    backup_path = write_yaml_with_backup(args.calibration_path, config)
    if backup_path is not None:
        print(f"Backup written: {backup_path}")
    print(f"Calibration updated: {args.calibration_path}")
    return 0


def run_jog_mode(args: argparse.Namespace) -> int:
    if args.hand == "both":
        raise ValueError("--jog requires --hand left or --hand right")
    if not sys.stdin.isatty():
        raise ValueError(
            "--jog needs an interactive terminal. Run `conda activate tv` first, then run "
            "`PYTHONNOUSERSITE=1 python calibrate_dex3_controller.py ... --jog` instead of `conda run`."
        )
    if args.jog_step <= 0.0:
        raise ValueError("--jog-step must be greater than zero")
    if args.jog_rate <= 0.0:
        raise ValueError("--jog-rate must be greater than zero")

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_

    hand = args.hand
    ChannelFactoryInitialize(0, networkInterface=args.interface)
    subscriber = ChannelSubscriber(state_topic(hand), HandState_)
    subscriber.Init()
    publisher = ChannelPublisher(command_topic(hand), HandCmd_)
    publisher.Init()
    msg = unitree_hg_msg_dds__HandCmd_()

    print("Stop teleop before jogging so only this helper commands Dex3.")
    print(f"Reading current {hand} Dex3 state...")
    target = read_hand_state_sample(subscriber, hand, args.timeout)
    selected_joint = JOINT_SPECS[hand][0][0]
    ranges = default_joint_ranges(hand)
    period = 1.0 / args.jog_rate
    print_jog_controls(hand, args.endpoint, selected_joint, target)

    save_means = None
    save_stddevs = None
    with RawTerminal():
        while True:
            start_time = time.monotonic()
            update_hand_command_msg(msg, hand, target, args.jog_kp, args.jog_kd)
            publisher.Write(msg)

            key = read_key()
            if key:
                numbered_joint = joint_key_by_number(hand, key)
                if numbered_joint is not None:
                    selected_joint = numbered_joint
                    print(f"\nSelected {selected_joint}")
                elif key in {"a", "A", "["}:
                    joint_range = ranges[selected_joint]
                    target[selected_joint] = clamp_angle(
                        target[selected_joint] - args.jog_step,
                        joint_range["lower"],
                        joint_range["upper"],
                    )
                    print(f"\r{selected_joint}: {target[selected_joint]: .4f} rad", end="", flush=True)
                elif key in {"d", "D", "]"}:
                    joint_range = ranges[selected_joint]
                    target[selected_joint] = clamp_angle(
                        target[selected_joint] + args.jog_step,
                        joint_range["lower"],
                        joint_range["upper"],
                    )
                    print(f"\r{selected_joint}: {target[selected_joint]: .4f} rad", end="", flush=True)
                elif key in {"z", "Z"}:
                    joint_range = ranges[selected_joint]
                    target[selected_joint] = step_toward(
                        target[selected_joint],
                        joint_range["open_angle"],
                        args.jog_step,
                    )
                    print(f"\r{selected_joint}: {target[selected_joint]: .4f} rad", end="", flush=True)
                elif key in {"x", "X"}:
                    joint_range = ranges[selected_joint]
                    target[selected_joint] = step_toward(
                        target[selected_joint],
                        joint_range["close_angle"],
                        args.jog_step,
                    )
                    print(f"\r{selected_joint}: {target[selected_joint]: .4f} rad", end="", flush=True)
                elif key in {"o", "O"}:
                    target[selected_joint] = ranges[selected_joint]["open_angle"]
                    print(f"\r{selected_joint}: {target[selected_joint]: .4f} rad", end="", flush=True)
                elif key in {"c", "C"}:
                    target[selected_joint] = ranges[selected_joint]["close_angle"]
                    print(f"\r{selected_joint}: {target[selected_joint]: .4f} rad", end="", flush=True)
                elif key in {"p", "P"}:
                    print_jog_targets(hand, selected_joint, target)
                elif key in {"s", "S"}:
                    print("\nCollecting measured state for save...")
                    samples = average_jog_state_samples(
                        subscriber,
                        publisher,
                        msg,
                        hand,
                        target,
                        args.samples,
                        args.timeout,
                        args.sample_period,
                        args.jog_kp,
                        args.jog_kd,
                    )
                    means, stddevs = summarize_samples(samples)
                    save_means = means
                    save_stddevs = stddevs
                    break
                elif key in {"q", "Q", "\x03"}:
                    print("\nJog mode closed without writing calibration.")
                    return 0

            elapsed = time.monotonic() - start_time
            time.sleep(max(0.0, period - elapsed))

    if save_means is None or save_stddevs is None:
        return 0
    return save_captured_endpoint(args, save_means, save_stddevs)


def summarize_samples(samples: dict[str, list[dict[str, float]]]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    means: dict[str, dict[str, float]] = {}
    stddevs: dict[str, dict[str, float]] = {}
    for hand, hand_samples in samples.items():
        means[hand] = {}
        stddevs[hand] = {}
        for joint_key, _, _ in JOINT_SPECS[hand]:
            values = [sample[joint_key] for sample in hand_samples]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            means[hand][joint_key] = mean
            stddevs[hand][joint_key] = math.sqrt(variance)
    return means, stddevs


def captured_raw_angles(means: dict[str, dict[str, float]]) -> tuple[dict[str, dict[str, float]], list[str]]:
    captured: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    for hand, joint_values in means.items():
        ranges = default_joint_ranges(hand)
        captured[hand] = {}
        for joint_key, current_q in joint_values.items():
            joint_range = ranges[joint_key]
            clamped_q = clamp_angle(current_q, joint_range["lower"], joint_range["upper"])
            captured[hand][joint_key] = current_q
            if not math.isclose(current_q, clamped_q, abs_tol=1e-6):
                warnings.append(
                    f"{hand}.{joint_key}: measured {current_q:.4f} rad is outside "
                    f"URDF limits [{joint_range['lower']:.4f}, {joint_range['upper']:.4f}]; "
                    f"runtime command will clamp to {clamped_q:.4f} rad"
                )
    return captured, warnings


def print_capture(
    endpoint: str,
    means: dict[str, dict[str, float]],
    stddevs: dict[str, dict[str, float]],
    captured: dict[str, dict[str, float]],
    max_std: float,
) -> list[str]:
    warnings: list[str] = []
    print(f"\nCaptured Dex3 endpoint: {endpoint}")
    print("hand   joint      raw_rad    std_rad   clamped_rad")
    print("-----  --------  --------  --------  -----------")
    for hand in ("left", "right"):
        if hand not in captured:
            continue
        ranges = default_joint_ranges(hand)
        for joint_key, _, _ in JOINT_SPECS[hand]:
            raw_q = means[hand][joint_key]
            std_q = stddevs[hand][joint_key]
            joint_range = ranges[joint_key]
            clamped_q = clamp_angle(raw_q, joint_range["lower"], joint_range["upper"])
            clamped_text = f"{clamped_q:11.4f}" if not math.isclose(raw_q, clamped_q, abs_tol=1e-6) else "          -"
            print(f"{hand:<5}  {joint_key:<8}  {raw_q:8.4f}  {std_q:8.4f}  {clamped_text}")
            if std_q > max_std:
                warnings.append(f"{hand}.{joint_key}: sample std {std_q:.4f} rad is above {max_std:.4f}")
    return warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teach Dex3 controller open_rad/grip_close_rad/trigger_close_rad calibration from current DDS state."
    )
    parser.add_argument("--interface", required=True, help="DDS network interface, for example eno1.")
    parser.add_argument("--hand", choices=["left", "right", "both"], default="both", help="Hand to capture.")
    parser.add_argument("--endpoint", choices=ENDPOINTS, required=True, help="Calibration endpoint to update.")
    parser.add_argument(
        "--calibration-path",
        type=Path,
        default=DEFAULT_CALIBRATION_PATH,
        help="YAML file to update.",
    )
    parser.add_argument("--samples", type=int, default=20, help="Number of DDS samples to average.")
    parser.add_argument("--timeout", type=float, default=10.0, help="Seconds to wait for DDS samples.")
    parser.add_argument("--sample-period", type=float, default=0.02, help="Seconds between DDS reads.")
    parser.add_argument("--max-std", type=float, default=0.03, help="Warn if sample std exceeds this many radians.")
    parser.add_argument("--dry-run", action="store_true", help="Print captured values without writing YAML.")
    parser.add_argument("-y", "--yes", action="store_true", help="Write without interactive confirmation.")
    parser.add_argument("--jog", action="store_true", help="Keyboard-jog one hand before saving the endpoint.")
    parser.add_argument("--jog-step", type=float, default=0.03, help="Jog step size in radians.")
    parser.add_argument("--jog-rate", type=float, default=30.0, help="Jog command publish rate in Hz.")
    parser.add_argument("--jog-kp", type=float, default=0.6, help="Jog position-control kp.")
    parser.add_argument("--jog-kd", type=float, default=0.2, help="Jog position-control kd.")
    return parser.parse_args()


def selected_hands(hand_arg: str) -> list[str]:
    if hand_arg == "both":
        return ["left", "right"]
    return [hand_arg]


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        print("--samples must be greater than zero", file=sys.stderr)
        return 2
    if args.timeout <= 0.0:
        print("--timeout must be greater than zero", file=sys.stderr)
        return 2
    if args.sample_period < 0.0:
        print("--sample-period must be zero or greater", file=sys.stderr)
        return 2

    hands = selected_hands(args.hand)

    try:
        if args.jog:
            return run_jog_mode(args)

        print("Stop teleop before capturing so nothing is commanding Dex3.")
        print(f"Move the {args.hand} hand to the desired {args.endpoint} pose, then wait for capture...")
        samples = collect_dds_samples(args.interface, hands, args.samples, args.timeout, args.sample_period)
        means, stddevs = summarize_samples(samples)
        return save_captured_endpoint(args, means, stddevs)
    except Exception as exc:
        print(f"Calibration failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
