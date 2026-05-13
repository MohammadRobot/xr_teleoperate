# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
import xml.etree.ElementTree as ET
from enum import IntEnum
import time
import os
import sys
import threading
from multiprocessing import Process, Array, Value, Lock

parent2_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent2_dir)
from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
from teleop.utils.weighted_moving_filter import WeightedMovingFilter

from teleop.utils.logging_compat import get_logger
logger_mp = get_logger(__name__)


Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(self, left_hand_array_in, right_hand_array_in, dual_hand_data_lock = None, dual_hand_state_array_out = None,
                       dual_hand_action_array_out = None, fps = 100.0, Unit_Test = False, simulation_mode = False,
                       manual_control = False, left_grip_value_in = None, right_grip_value_in = None,
                       grip_inverted = False, left_trigger_value_in = None, right_trigger_value_in = None,
                       left_squeeze_value_in = None, right_squeeze_value_in = None,
                       left_trigger_pressed_in = None, right_trigger_pressed_in = None,
                       left_squeeze_pressed_in = None, right_squeeze_pressed_in = None,
                       manual_deadzone = 0.05, manual_kp = 0.8, manual_kp_boost = 1.2, manual_kd = 0.2,
                       controller_calibration_path = None):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_array_in: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_array_in: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array_out: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array_out: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """
        logger_mp.info("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode
        self.manual_control = manual_control
        self._left_grip_value_in = left_grip_value_in
        self._right_grip_value_in = right_grip_value_in
        self._left_trigger_value_in = left_trigger_value_in
        self._right_trigger_value_in = right_trigger_value_in
        self._left_squeeze_value_in = left_squeeze_value_in
        self._right_squeeze_value_in = right_squeeze_value_in
        self._left_trigger_pressed_in = left_trigger_pressed_in
        self._right_trigger_pressed_in = right_trigger_pressed_in
        self._left_squeeze_pressed_in = left_squeeze_pressed_in
        self._right_squeeze_pressed_in = right_squeeze_pressed_in
        self._grip_inverted = grip_inverted
        self._manual_deadzone = manual_deadzone
        self._manual_kp = manual_kp
        self._manual_kp_boost = manual_kp_boost
        self._manual_kd = manual_kd
        self._controller_calibration_path = controller_calibration_path
        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)
        else:
            self.hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3_Unit_Test)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        if self.manual_control:
            self._left_joint_names = [
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "left_hand_middle_0_joint",
                "left_hand_middle_1_joint",
                "left_hand_index_0_joint",
                "left_hand_index_1_joint",
            ]
            self._right_joint_names = [
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint",
                "right_hand_index_0_joint",
                "right_hand_index_1_joint",
            ]
            self._finger_joint_indices = {
                "left": {
                    "thumb": [0, 1, 2],
                    "middle": [3, 4],
                    "index": [5, 6],
                },
                "right": {
                    "thumb": [0, 1, 2],
                    "middle": [3, 4],
                    "index": [5, 6],
                },
            }
            self._joint_calibration_keys = {
                "left": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
                "right": ["thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1"],
            }
            self._controller_calibration = self._load_dex3_controller_calibration(self._controller_calibration_path)
            self._left_open, self._left_grip_close, self._left_trigger_close = self._load_dex3_joint_targets(hand="left")
            self._right_open, self._right_grip_close, self._right_trigger_close = self._load_dex3_joint_targets(hand="right")
            self._left_close = self._left_grip_close
            self._right_close = self._right_grip_close

        hand_control_process = Process(target=self.control_process, args=(left_hand_array_in, right_hand_array_in,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array_out, dual_hand_action_array_out))
        hand_control_process.daemon = True
        hand_control_process.start()

        logger_mp.info("Initialize Dex3_1_Controller OK!")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target, left_boost_mask=None, right_boost_mask=None):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
            if left_boost_mask is not None:
                self.left_msg.motor_cmd[id].kp = self._manual_kp_boost if left_boost_mask[idx] else self._manual_kp
                self.left_msg.motor_cmd[id].kd = self._manual_kd
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]
            if right_boost_mask is not None:
                self.right_msg.motor_cmd[id].kp = self._manual_kp_boost if right_boost_mask[idx] else self._manual_kp
                self.right_msg.motor_cmd[id].kd = self._manual_kd

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)
        # logger_mp.debug("hand ctrl publish ok.")
    
    def control_process(self, left_hand_array_in, right_hand_array_in, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array_out = None, dual_hand_action_array_out = None):
        self.running = True

        left_q_target  = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = self._manual_kp
        kd = self._manual_kd

        # initialize dex3-1's left hand cmd msg
        self.left_msg  = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q    = q
            self.left_msg.motor_cmd[id].dq   = dq
            self.left_msg.motor_cmd[id].tau  = tau
            self.left_msg.motor_cmd[id].kp   = kp
            self.left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id = id, status = 0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode  
            self.right_msg.motor_cmd[id].q    = q
            self.right_msg.motor_cmd[id].dq   = dq
            self.right_msg.motor_cmd[id].tau  = tau
            self.right_msg.motor_cmd[id].kp   = kp
            self.right_msg.motor_cmd[id].kd   = kd  

        try:
            while self.running:
                start_time = time.time()
                left_boost_mask = None
                right_boost_mask = None
                if self.manual_control:
                    left_trigger, left_squeeze, left_trigger_pressed, left_squeeze_pressed = self._read_manual_inputs("left")
                    right_trigger, right_squeeze, right_trigger_pressed, right_squeeze_pressed = self._read_manual_inputs("right")
                    left_q_target = self._manual_grasp_target("left", left_trigger, left_squeeze)
                    right_q_target = self._manual_grasp_target("right", right_trigger, right_squeeze)
                    left_boost_mask = self._manual_boost_mask("left", left_trigger_pressed, left_squeeze_pressed)
                    right_boost_mask = self._manual_boost_mask("right", right_trigger_pressed, right_squeeze_pressed)
                else:
                    # get dual hand state
                    with left_hand_array_in.get_lock():
                        left_hand_data  = np.array(left_hand_array_in[:]).reshape(25, 3).copy()
                    with right_hand_array_in.get_lock():
                        right_hand_data = np.array(right_hand_array_in[:]).reshape(25, 3).copy()

                    if not np.all(right_hand_data == 0.0) and not np.all(left_hand_data[4] == np.array([-1.13, 0.3, 0.15])): # if hand data has been initialized.
                        ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                        ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                        left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                        right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array_out and dual_hand_action_array_out:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_action_array_out[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target, left_boost_mask, right_boost_mask)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex3_1_Controller has been closed.")

    def _interp_grip(self, grip: float, open_vals: np.ndarray, close_vals: np.ndarray) -> np.ndarray:
        grip = float(np.clip(grip, 0.0, 1.0))
        if self._grip_inverted:
            grip = 1.0 - grip
        return open_vals + grip * (close_vals - open_vals)

    def _read_shared_value(self, shared_value) -> float:
        if shared_value is None:
            return 0.0
        with shared_value.get_lock():
            return float(shared_value.value)

    def _read_shared_bool(self, shared_value) -> bool:
        if shared_value is None:
            return False
        with shared_value.get_lock():
            return bool(shared_value.value)

    def _normalize_manual_value(self, value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        if self._grip_inverted:
            value = 1.0 - value
        if value < self._manual_deadzone:
            return 0.0
        return value

    def _read_manual_inputs(self, hand: str) -> tuple[float, float, bool, bool]:
        if hand == "left":
            trigger_value = self._read_shared_value(self._left_trigger_value_in)
            squeeze_value = self._read_shared_value(self._left_squeeze_value_in)
            grip_value = self._read_shared_value(self._left_grip_value_in)
            trigger_pressed = self._read_shared_bool(self._left_trigger_pressed_in)
            squeeze_pressed = self._read_shared_bool(self._left_squeeze_pressed_in)
        else:
            trigger_value = self._read_shared_value(self._right_trigger_value_in)
            squeeze_value = self._read_shared_value(self._right_squeeze_value_in)
            grip_value = self._read_shared_value(self._right_grip_value_in)
            trigger_pressed = self._read_shared_bool(self._right_trigger_pressed_in)
            squeeze_pressed = self._read_shared_bool(self._right_squeeze_pressed_in)

        if self._left_trigger_value_in is None and self._right_trigger_value_in is None and (
            self._left_grip_value_in is not None or self._right_grip_value_in is not None
        ):
            trigger_value = 0.0
            squeeze_value = grip_value
            trigger_pressed = False
            squeeze_pressed = grip_value >= self._manual_deadzone

        return (
            self._normalize_manual_value(trigger_value),
            self._normalize_manual_value(squeeze_value),
            trigger_pressed,
            squeeze_pressed,
        )

    def _manual_grasp_target(self, hand: str, trigger: float, squeeze: float) -> np.ndarray:
        if hand == "left":
            open_vals = self._left_open
            grip_close_vals = self._left_grip_close
            trigger_close_vals = self._left_trigger_close
            trigger_mask = np.array([True, True, True, False, False, True, True], dtype=bool)
        else:
            open_vals = self._right_open
            grip_close_vals = self._right_grip_close
            trigger_close_vals = self._right_trigger_close
            trigger_mask = np.array([True, True, True, False, False, True, True], dtype=bool)
        if squeeze > 0.0:
            return open_vals + squeeze * (grip_close_vals - open_vals)
        close_levels = np.zeros(Dex3_Num_Motors, dtype=np.float64)
        close_levels[trigger_mask] = trigger
        return open_vals + close_levels * (trigger_close_vals - open_vals)

    def _manual_boost_mask(self, hand: str, trigger_pressed: bool, squeeze_pressed: bool) -> np.ndarray:
        if squeeze_pressed:
            return np.ones(Dex3_Num_Motors, dtype=bool)
        if hand == "left":
            return np.array([trigger_pressed, trigger_pressed, trigger_pressed, False, False, trigger_pressed, trigger_pressed], dtype=bool)
        return np.array([trigger_pressed, trigger_pressed, trigger_pressed, False, False, trigger_pressed, trigger_pressed], dtype=bool)

    def _load_dex3_joint_targets(self, hand: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        joint_names = self._left_joint_names if hand == "left" else self._right_joint_names
        limits = self._load_dex3_joint_limits(hand)
        open_vals = []
        close_vals = []
        lower_vals = []
        upper_vals = []
        for name in joint_names:
            lower, upper = limits.get(name, (0.0, 0.0))
            open_val = float(np.clip(0.0, lower, upper))
            if upper > 0.0:
                close_val = upper
            else:
                close_val = lower
            close_val = float(np.clip(close_val, lower, upper))
            open_vals.append(open_val)
            close_vals.append(close_val)
            lower_vals.append(lower)
            upper_vals.append(upper)
        open_vals = np.array(open_vals, dtype=np.float64)
        close_vals = np.array(close_vals, dtype=np.float64)
        lower_vals = np.array(lower_vals, dtype=np.float64)
        upper_vals = np.array(upper_vals, dtype=np.float64)
        return self._apply_dex3_controller_calibration(hand, open_vals, close_vals, close_vals.copy(), lower_vals, upper_vals)

    def _apply_dex3_controller_calibration(
        self,
        hand: str,
        open_vals: np.ndarray,
        grip_close_vals: np.ndarray,
        trigger_close_vals: np.ndarray,
        lower_vals: np.ndarray,
        upper_vals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not getattr(self, "_controller_calibration", None):
            return open_vals, grip_close_vals, trigger_close_vals
        calibrated_open = open_vals.copy()
        calibrated_grip_close = grip_close_vals.copy()
        calibrated_trigger_close = trigger_close_vals.copy()
        hand_config = self._controller_calibration.get(hand, {})
        joint_config = hand_config.get("joints")
        if isinstance(joint_config, dict):
            for idx, joint_key in enumerate(self._joint_calibration_keys[hand]):
                endpoint_config = joint_config.get(joint_key, {})
                calibrated_open[idx] = self._dex3_endpoint_target(
                    endpoint_config, "open", idx, open_vals, grip_close_vals, lower_vals, upper_vals, f"{hand}.joints.{joint_key}"
                )
                calibrated_grip_close[idx] = self._dex3_endpoint_target(
                    endpoint_config, "grip_close", idx, open_vals, grip_close_vals, lower_vals, upper_vals, f"{hand}.joints.{joint_key}"
                )
                calibrated_trigger_close[idx] = self._dex3_endpoint_target(
                    endpoint_config, "trigger_close", idx, open_vals, grip_close_vals, lower_vals, upper_vals, f"{hand}.joints.{joint_key}"
                )
            return calibrated_open, calibrated_grip_close, calibrated_trigger_close
        for finger, indices in self._finger_joint_indices[hand].items():
            finger_config = hand_config.get(finger, {})
            for idx in indices:
                calibrated_open[idx] = self._dex3_endpoint_target(
                    finger_config, "open", idx, open_vals, grip_close_vals, lower_vals, upper_vals, f"{hand}.{finger}"
                )
                calibrated_grip_close[idx] = self._dex3_endpoint_target(
                    finger_config, "grip_close", idx, open_vals, grip_close_vals, lower_vals, upper_vals, f"{hand}.{finger}"
                )
                calibrated_trigger_close[idx] = self._dex3_endpoint_target(
                    finger_config, "trigger_close", idx, open_vals, grip_close_vals, lower_vals, upper_vals, f"{hand}.{finger}"
                )
        return calibrated_open, calibrated_grip_close, calibrated_trigger_close

    def _dex3_endpoint_target(
        self,
        endpoint_config: dict,
        endpoint: str,
        idx: int,
        open_vals: np.ndarray,
        close_vals: np.ndarray,
        lower_vals: np.ndarray,
        upper_vals: np.ndarray,
        label: str,
    ) -> float:
        raw_key = f"{endpoint}_rad"
        if raw_key in endpoint_config:
            raw_value = float(endpoint_config[raw_key])
            return self._clamp_dex3_raw_target(raw_value, lower_vals[idx], upper_vals[idx], f"{label}.{raw_key}")

        if endpoint == "open":
            fraction = endpoint_config.get("open", 0.0)
        elif endpoint == "grip_close":
            fraction = endpoint_config.get("grip_close", endpoint_config.get("close", 1.0))
        elif endpoint == "trigger_close":
            fraction = endpoint_config.get("trigger_close", endpoint_config.get("close", 1.0))
        else:
            raise ValueError(f"unknown Dex3 endpoint: {endpoint}")
        fraction = float(np.clip(float(fraction), 0.0, 1.0))
        target = open_vals[idx] + fraction * (close_vals[idx] - open_vals[idx])
        return float(np.clip(target, lower_vals[idx], upper_vals[idx]))

    def _clamp_dex3_raw_target(self, raw_value: float, lower: float, upper: float, label: str) -> float:
        clamped_value = float(np.clip(raw_value, lower, upper))
        if not np.isclose(raw_value, clamped_value, atol=1e-6):
            logger_mp.warning(
                f"[Dex3_1_Controller] {label}={raw_value:.4f} rad is outside URDF limits "
                f"[{lower:.4f}, {upper:.4f}]; using {clamped_value:.4f} rad."
            )
        return clamped_value

    def _default_dex3_controller_calibration(self) -> dict:
        return {
            hand: {
                finger: {"open": 0.0, "grip_close": 1.0, "trigger_close": 1.0}
                for finger in ("thumb", "index", "middle")
            }
            for hand in ("left", "right")
        }

    def _load_dex3_controller_calibration(self, calibration_path = None) -> dict:
        default_config = self._default_dex3_controller_calibration()
        if calibration_path is None:
            calibration_path = os.path.join(parent2_dir, "assets", "unitree_hand", "dex3_controller_calibration.yml")
        if not os.path.exists(calibration_path):
            logger_mp.warning(f"[Dex3_1_Controller] Dex3 controller calibration not found: {calibration_path}; using URDF limits.")
            return default_config
        try:
            import yaml
            with open(calibration_path, "r", encoding="utf-8") as f:
                loaded_config = yaml.safe_load(f)
            if loaded_config is None:
                loaded_config = {}
            if not isinstance(loaded_config, dict):
                raise ValueError("top-level YAML value must be a mapping")
            config = self._default_dex3_controller_calibration()
            for hand in ("left", "right"):
                hand_config = loaded_config.get(hand, {})
                if hand_config is None:
                    hand_config = {}
                if not isinstance(hand_config, dict):
                    raise ValueError(f"{hand} must be a mapping")
                joint_config = hand_config.get("joints")
                if joint_config is not None:
                    if not isinstance(joint_config, dict):
                        raise ValueError(f"{hand}.joints must be a mapping")
                    config[hand] = {"joints": {}}
                    for joint_key in self._joint_calibration_keys[hand]:
                        config[hand]["joints"][joint_key] = self._parse_dex3_calibration_endpoint(
                            joint_config.get(joint_key, {}),
                            f"{hand}.joints.{joint_key}",
                        )
                    continue
                for finger in ("thumb", "index", "middle"):
                    config[hand][finger] = self._parse_dex3_calibration_endpoint(
                        hand_config.get(finger, {}),
                        f"{hand}.{finger}",
                    )
            return config
        except Exception as e:
            logger_mp.warning(f"[Dex3_1_Controller] Failed to load Dex3 controller calibration {calibration_path}: {e}; using URDF limits.")
            return default_config

    def _parse_dex3_calibration_endpoint(self, endpoint_config, label: str) -> dict:
        if endpoint_config is None:
            endpoint_config = {}
        if not isinstance(endpoint_config, dict):
            raise ValueError(f"{label} must be a mapping")
        return {
            "open": float(np.clip(float(endpoint_config.get("open", 0.0)), 0.0, 1.0)),
            "grip_close": float(np.clip(float(endpoint_config.get("grip_close", endpoint_config.get("close", 1.0))), 0.0, 1.0)),
            "trigger_close": float(np.clip(float(endpoint_config.get("trigger_close", endpoint_config.get("close", 1.0))), 0.0, 1.0)),
            **self._parse_dex3_raw_calibration_fields(endpoint_config),
        }

    def _parse_dex3_raw_calibration_fields(self, endpoint_config) -> dict:
        parsed = {}
        for raw_key in ("open_rad", "grip_close_rad", "trigger_close_rad"):
            if raw_key in endpoint_config:
                parsed[raw_key] = float(endpoint_config[raw_key])
        return parsed

    def _load_dex3_joint_limits(self, hand: str) -> dict:
        base_dir = os.path.join(parent2_dir, "assets", "unitree_hand")
        if hand == "left":
            urdf_path = os.path.join(base_dir, "unitree_dex3_left.urdf")
        else:
            urdf_path = os.path.join(base_dir, "unitree_dex3_right.urdf")
        limits = {}
        try:
            tree = ET.parse(urdf_path)
            root = tree.getroot()
            for joint in root.findall("joint"):
                name = joint.get("name")
                limit = joint.find("limit")
                if name is None or limit is None:
                    continue
                try:
                    lower = float(limit.get("lower", "0"))
                    upper = float(limit.get("upper", "0"))
                except Exception:
                    continue
                limits[name] = (lower, upper)
        except Exception as e:
            logger_mp.warning(f"[Dex3_1_Controller] Failed to parse dex3 urdf limits: {e}")
        return limits

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandMiddle0 = 3
    kRightHandMiddle1 = 4
    kRightHandIndex0 = 5
    kRightHandIndex1 = 6


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"

class Dex1_1_Gripper_Controller:
    def __init__(self, left_gripper_value_in, right_gripper_value_in, dual_gripper_data_lock = None, dual_gripper_state_out = None, dual_gripper_action_out = None, 
                       filter = True, fps = 200.0, Unit_Test = False, simulation_mode = False):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_gripper_value_in: [input] Left ctrl data (required from XR device) to control_thread

        right_gripper_value_in: [input] Right ctrl data (required from XR device) to control_thread

        dual_gripper_data_lock: Data synchronization lock for dual_gripper_state_array and dual_gripper_action_array

        dual_gripper_state_out: [output] Return left(1), right(1) gripper motor state

        dual_gripper_action_out: [output] Return left(1), right(1) gripper motor action

        fps: Control frequency

        Unit_Test: Whether to enable unit testing

        simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """

        logger_mp.info("Initialize Dex1_1_Gripper_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.gripper_sub_ready = False
        self.simulation_mode = simulation_mode
        
        if filter and not self.simulation_mode:
            self.smooth_filter = WeightedMovingFilter(np.array([0.5, 0.3, 0.2]), 2)
        else:
            self.smooth_filter = None
 
        # initialize handcmd publisher and handstate subscriber
        self.LeftGripperCmb_publisher = ChannelPublisher(kTopicGripperLeftCommand, MotorCmds_)
        self.LeftGripperCmb_publisher.Init()
        self.RightGripperCmb_publisher = ChannelPublisher(kTopicGripperRightCommand, MotorCmds_)
        self.RightGripperCmb_publisher.Init()

        self.LeftGripperState_subscriber = ChannelSubscriber(kTopicGripperLeftState, MotorStates_)
        self.LeftGripperState_subscriber.Init()
        self.RightGripperState_subscriber = ChannelSubscriber(kTopicGripperRightState, MotorStates_)
        self.RightGripperState_subscriber.Init()

        # Shared Arrays for gripper states
        self.left_gripper_state_value = Value('d', 0.0, lock=True)
        self.right_gripper_state_value = Value('d', 0.0, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_gripper_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while not self.gripper_sub_ready:
            time.sleep(0.01)
            logger_mp.warning("[Dex1_1_Gripper_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex1_1_Gripper_Controller] Subscribe dds ok.")

        self.gripper_control_thread = threading.Thread(target=self.control_thread, args=(left_gripper_value_in, right_gripper_value_in, self.left_gripper_state_value, self.right_gripper_state_value,
                                                                                         dual_gripper_data_lock, dual_gripper_state_out, dual_gripper_action_out))
        self.gripper_control_thread.daemon = True
        self.gripper_control_thread.start()

        logger_mp.info("Initialize Dex1_1_Gripper_Controller OK!")

    def _subscribe_gripper_state(self):
        while True:
            left_gripper_msg  = self.LeftGripperState_subscriber.Read()
            right_gripper_msg  = self.RightGripperState_subscriber.Read()
            self.gripper_sub_ready = True
            if left_gripper_msg is not None and right_gripper_msg is not None:
                self.left_gripper_state_value.value = left_gripper_msg.states[0].q
                self.right_gripper_state_value.value = right_gripper_msg.states[0].q
            time.sleep(0.002)
    
    def ctrl_dual_gripper(self, dual_gripper_action):
        """set current left, right gripper motor cmd target q"""
        self.left_gripper_msg.cmds[0].q  = dual_gripper_action[0]
        self.right_gripper_msg.cmds[0].q = dual_gripper_action[1]

        self.LeftGripperCmb_publisher.Write(self.left_gripper_msg)
        self.RightGripperCmb_publisher.Write(self.right_gripper_msg)
        # logger_mp.debug("gripper ctrl publish ok.")
    
    def control_thread(self, left_gripper_value_in, right_gripper_value_in, left_gripper_state_value, right_gripper_state_value, dual_hand_data_lock = None, 
                             dual_gripper_state_out = None, dual_gripper_action_out = None):
        self.running = True
        DELTA_GRIPPER_CMD = 0.18     # The motor rotates 5.4 radians, the clamping jaw slide open 9 cm, so 0.6 rad <==> 1 cm, 0.18 rad <==> 3 mm
        THUMB_INDEX_DISTANCE_MIN = 5.0
        THUMB_INDEX_DISTANCE_MAX = 7.0
        LEFT_MAPPED_MIN  = 0.0           # The minimum initial motor position when the gripper closes at startup.
        RIGHT_MAPPED_MIN = 0.0           # The minimum initial motor position when the gripper closes at startup.
        # The maximum initial motor position when the gripper closes before calibration (with the rail stroke calculated as 0.6 cm/rad * 9 rad = 5.4 cm).
        LEFT_MAPPED_MAX = LEFT_MAPPED_MIN + 5.40 
        RIGHT_MAPPED_MAX = RIGHT_MAPPED_MIN + 5.40
        left_target_action  = (LEFT_MAPPED_MAX - LEFT_MAPPED_MIN) / 2.0
        right_target_action = (RIGHT_MAPPED_MAX - RIGHT_MAPPED_MIN) / 2.0

        dq = 0.0
        tau = 0.0
        kp = 5.00
        kd = 0.05
        # initialize gripper cmd msg
        self.left_gripper_msg  = MotorCmds_()
        self.left_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]
        self.right_gripper_msg = MotorCmds_()
        self.right_gripper_msg.cmds = [unitree_go_msg_dds__MotorCmd_()]

        self.left_gripper_msg.cmds[0].dq  = dq
        self.left_gripper_msg.cmds[0].tau = tau
        self.left_gripper_msg.cmds[0].kp  = kp
        self.left_gripper_msg.cmds[0].kd  = kd

        self.right_gripper_msg.cmds[0].dq  = dq
        self.right_gripper_msg.cmds[0].tau = tau
        self.right_gripper_msg.cmds[0].kp  = kp
        self.right_gripper_msg.cmds[0].kd  = kd
        try:
            while self.running:
                start_time = time.time()
                # get dual hand skeletal point state from XR device
                with left_gripper_value_in.get_lock():
                    left_gripper_value  = left_gripper_value_in.value
                with right_gripper_value_in.get_lock():
                    right_gripper_value = right_gripper_value_in.value
                # get current dual gripper motor state
                dual_gripper_state = np.array([left_gripper_state_value.value, right_gripper_state_value.value])
                
                if left_gripper_value != 0.0 or right_gripper_value != 0.0: # if input data has been initialized.
                    # Linear mapping from [0, THUMB_INDEX_DISTANCE_MAX] to gripper action range
                    left_target_action  = np.interp(left_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [LEFT_MAPPED_MIN, LEFT_MAPPED_MAX])
                    right_target_action = np.interp(right_gripper_value, [THUMB_INDEX_DISTANCE_MIN, THUMB_INDEX_DISTANCE_MAX], [RIGHT_MAPPED_MIN, RIGHT_MAPPED_MAX])
                # clip dual gripper action to avoid overflow
                if not self.simulation_mode:
                    left_actual_action  = np.clip(left_target_action,  dual_gripper_state[0] - DELTA_GRIPPER_CMD, dual_gripper_state[0] + DELTA_GRIPPER_CMD) 
                    right_actual_action = np.clip(right_target_action, dual_gripper_state[1] - DELTA_GRIPPER_CMD, dual_gripper_state[1] + DELTA_GRIPPER_CMD)
                else:
                    left_actual_action  = left_target_action
                    right_actual_action = right_target_action
                dual_gripper_action = np.array([left_actual_action, right_actual_action])

                if self.smooth_filter:
                    self.smooth_filter.add_data(dual_gripper_action)
                    dual_gripper_action = self.smooth_filter.filtered_data

                if dual_gripper_state_out and dual_gripper_action_out:
                    with dual_hand_data_lock:
                        dual_gripper_state_out[:] = dual_gripper_state - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])
                        dual_gripper_action_out[:] = dual_gripper_action - np.array([LEFT_MAPPED_MIN, RIGHT_MAPPED_MIN])

                self.ctrl_dual_gripper(dual_gripper_action)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Dex1_1_Gripper_Controller has been closed.")

class Gripper_JointIndex(IntEnum):
    kGripper = 0


if __name__ == "__main__":
    import argparse
    from televuer import TeleVuerWrapper
    from teleimager import ImageClient

    parser = argparse.ArgumentParser()
    parser.add_argument('--xr-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device tracking source')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire1', 'brainco'], help='Select end effector controller')
    args = parser.parse_args()
    logger_mp.info(f"args:{args}\n")

    ChannelFactoryInitialize(1) # 0 for real robot, 1 for simulation
    
    # image client
    img_client = ImageClient(host='127.0.0.1') #host='192.168.123.164'
    if not img_client.has_head_cam():
        logger_mp.error("Head camera is required. Please enable head camera on the image server side.")
    head_img_shape = img_client.get_head_shape()
    tv_binocular = img_client.head_is_binocular()

    # television: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
    tv_wrapper = TeleVuerWrapper(binocular=tv_binocular, use_hand_tracking=args.xr_mode == "hand", img_shape=head_img_shape, return_hand_rot_data = False)

# end-effector
    if args.ee == "dex3":
        left_hand_pos_array = Array('d', 75, lock = True)      # [input]
        right_hand_pos_array = Array('d', 75, lock = True)     # [input]
        dual_hand_data_lock = Lock()
        dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
        dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
        hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array)
    elif args.ee == "dex1":
        left_gripper_value = Value('d', 0.0, lock=True)        # [input]
        right_gripper_value = Value('d', 0.0, lock=True)       # [input]
        dual_gripper_data_lock = Lock()
        dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
        dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
        gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, dual_gripper_state_array, dual_gripper_action_array)

    user_input = input("Please enter the start signal (enter 's' to start the subsequent program):\n")
    if user_input.lower() == 's':
        while True:
            head_img, head_img_fps = img_client.get_head_frame()
            tv_wrapper.set_display_image(head_img)
            tele_data = tv_wrapper.get_tele_data()
            if args.ee == "dex3" and args.xr_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex1" and args.xr_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.xr_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass

            # with dual_hand_data_lock:
            #     logger_mp.info(f"state : {list(dual_hand_state_array)} \naction: {list(dual_hand_action_array)} \n")
            time.sleep(0.01)
