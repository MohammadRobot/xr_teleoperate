import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.utils.logging_compat import logging_mp, basic_config, get_logger
basic_config(level=logging_mp.INFO)
logger_mp = get_logger(__name__)

def publish_reset_category(category: int, publisher): # Scene Reset signal
    from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.

def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START == True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")

def apply_sync_toggle(sync_enabled: bool, button_pressed: bool, previous_button_pressed: bool) -> tuple[bool, bool, bool]:
    """Toggle sync once on a controller button rising edge."""
    button_pressed = bool(button_pressed)
    toggled = button_pressed and not previous_button_pressed
    if toggled:
        sync_enabled = not sync_enabled
    return sync_enabled, button_pressed, toggled

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['none', 'dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], default='dex3', help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    parser.add_argument('--motion', action = 'store_true', help = 'Enable motion control mode')
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--no-camera', action='store_true', help='Disable Tele Imager/WebRTC image input and run XR in pass-through mode only')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')
    parser.add_argument('--dex3-kp', type=float, default=0.8, help='Dex3 controller-mode normal position stiffness.')
    parser.add_argument('--dex3-kp-boost', type=float, default=1.2, help='Dex3 controller-mode boosted position stiffness when trigger/grip is pressed.')
    parser.add_argument('--dex3-kd', type=float, default=0.2, help='Dex3 controller-mode position damping.')

    args = parser.parse_args()
    if args.no_camera and args.display_mode != 'pass-through':
        parser.error('--no-camera requires --display-mode=pass-through because no image source is available.')
    if args.no_camera and args.record:
        parser.error('--no-camera cannot be used with --record because recording currently expects camera frames.')
    if args.dex3_kp < 0.0 or args.dex3_kp_boost < 0.0 or args.dex3_kd < 0.0:
        parser.error('--dex3-kp, --dex3-kp-boost, and --dex3-kd must be non-negative.')
    logger_mp.info(f"args: {args}")

    motion_switcher = None
    loco_wrapper = None
    img_client = None
    tv_wrapper = None
    arm_ctrl = None
    recorder = None
    ipc_server = None
    listen_keyboard_thread = None
    sim_state_subscriber = None
    stop_listening_fn = None

    try:
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher # dds
        from televuer import TeleVuerWrapper
        from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
        from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
        from teleop.utils.episode_writer import EpisodeWriter
        from teleop.utils.ipc import IPC_Server
        from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
        from sshkeyboard import listen_keyboard, stop_listening as stop_listening_fn

        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        if args.no_camera:
            logger_mp.info("No-camera mode enabled: skipping Tele Imager/WebRTC and using XR pass-through.")
            camera_config = {
                'head_camera': {
                    'enable_zmq': False,
                    'enable_webrtc': False,
                    'image_shape': (480, 640),
                    'binocular': False,
                    'fps': args.frequency,
                    'webrtc_port': None,
                },
                'left_wrist_camera': {'enable_zmq': False},
                'right_wrist_camera': {'enable_zmq': False},
            }
            xr_need_local_img = False
            tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand",
                                         binocular=False,
                                         img_shape=(480, 640),
                                         display_fps=args.frequency,
                                         display_mode='pass-through',
                                         zmq=False,
                                         webrtc=False)
        else:
            from teleimager.image_client import ImageClient

            # image client
            img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
            camera_config = img_client.get_cam_config()
            logger_mp.debug(f"Camera config: {camera_config}")
            xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

            # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
            tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand",
                                         binocular=camera_config['head_camera']['binocular'],
                                         img_shape=camera_config['head_camera']['image_shape'],
                                         # maybe should decrease fps for better performance?
                                         # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                         # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                         display_mode=args.display_mode,
                                         zmq=camera_config['head_camera']['enable_zmq'],
                                         webrtc=camera_config['head_camera']['enable_webrtc'],
                                         webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                         )
        
        # motion mode (G1: Regular mode R1+X, not Running mode R2+A)
        if args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            if args.sim:
                logger_mp.info("Simulation mode: skip motion switcher debug mode.")
            else:
                motion_switcher = MotionSwitcher()
                status, result = motion_switcher.Enter_Debug_Mode()
                logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        if args.arm == "G1_29":
            arm_ik = G1_29_ArmIK()
            arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "G1_23":
            arm_ik = G1_23_ArmIK()
            arm_ctrl = G1_23_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1_2":
            arm_ik = H1_2_ArmIK()
            arm_ctrl = H1_2_ArmController(motion_mode=args.motion, simulation_mode=args.sim)
        elif args.arm == "H1":
            arm_ik = H1_ArmIK()
            arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            if args.input_mode == "controller":
                left_dex3_trigger_value = Value('d', 0.0, lock=True)     # [input] thumb + index pinch
                right_dex3_trigger_value = Value('d', 0.0, lock=True)    # [input] thumb + index pinch
                left_dex3_squeeze_value = Value('d', 0.0, lock=True)     # [input] full-hand grip
                right_dex3_squeeze_value = Value('d', 0.0, lock=True)    # [input] full-hand grip
                left_dex3_trigger_pressed = Value('b', False, lock=True)  # [input] boost thumb + index pinch
                right_dex3_trigger_pressed = Value('b', False, lock=True) # [input] boost thumb + index pinch
                left_dex3_squeeze_pressed = Value('b', False, lock=True)  # [input] boost full-hand grip
                right_dex3_squeeze_pressed = Value('b', False, lock=True) # [input] boost full-hand grip
                hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                              dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim,
                                              manual_control=True,
                                              left_trigger_value_in=left_dex3_trigger_value,
                                              right_trigger_value_in=right_dex3_trigger_value,
                                              left_squeeze_value_in=left_dex3_squeeze_value,
                                              right_squeeze_value_in=right_dex3_squeeze_value,
                                              left_trigger_pressed_in=left_dex3_trigger_pressed,
                                              right_trigger_pressed_in=right_dex3_trigger_pressed,
                                              left_squeeze_pressed_in=left_dex3_squeeze_pressed,
                                              right_squeeze_pressed_in=right_dex3_squeeze_pressed,
                                              manual_kp=args.dex3_kp,
                                              manual_kp_boost=args.dex3_kp_boost,
                                              manual_kd=args.dex3_kd)
            else:
                hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                              dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_ftp":
            from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            recorder = EpisodeWriter(task_dir = os.path.join(args.task_dir, args.task_name),
                                     task_goal = args.task_goal,
                                     task_desc = args.task_desc,
                                     task_steps = args.task_steps,
                                     frequency = args.frequency, 
                                     rerun_log = not args.headless)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("Press [r] or left controller [Y/B] to start syncing the robot with your movements.")
        logger_mp.info("Press left controller [Y/B] again to pause or resume arm sync.")
        if args.record:
            logger_mp.info("Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("Recording is DISABLED (run with --record to enable).")
        logger_mp.info("Press [q] or right controller [A] to stop and exit the program.")
        logger_mp.info("IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        left_sync_button_was_pressed = False
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)
            if args.input_mode == "controller":
                tele_data = tv_wrapper.get_tele_data()
                if tele_data.right_ctrl_aButton:
                    STOP = True
                    break
                START, left_sync_button_was_pressed, toggled = apply_sync_toggle(
                    START,
                    tele_data.left_ctrl_bButton,
                    left_sync_button_was_pressed,
                )
                if toggled and START:
                    logger_mp.info("Arm sync enabled from left controller [Y/B].")

        if STOP:
            logger_mp.info("Exit requested before robot sync started.")
            raise SystemExit(0)
        logger_mp.info("---------------------start Tracking-------------------------")
        arm_ctrl.speed_gradual_max()
        sync_enabled_last = START
        held_arm_q = arm_ctrl.get_current_dual_arm_q().copy()
        held_arm_tauff = held_arm_q * 0.0
        # main loop. robot start to follow VR user's motion
        while not STOP:
            start_time = time.time()
            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            if (args.ee == "dex3" or args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex3" and args.input_mode == "controller":
                left_trigger = max(0.0, min(1.0, (10.0 - tele_data.left_ctrl_triggerValue) / 10.0))
                right_trigger = max(0.0, min(1.0, (10.0 - tele_data.right_ctrl_triggerValue) / 10.0))
                left_squeeze = max(0.0, min(1.0, tele_data.left_ctrl_squeezeValue))
                right_squeeze = max(0.0, min(1.0, tele_data.right_ctrl_squeezeValue))
                with left_dex3_trigger_value.get_lock():
                    left_dex3_trigger_value.value = left_trigger
                with right_dex3_trigger_value.get_lock():
                    right_dex3_trigger_value.value = right_trigger
                with left_dex3_squeeze_value.get_lock():
                    left_dex3_squeeze_value.value = left_squeeze
                with right_dex3_squeeze_value.get_lock():
                    right_dex3_squeeze_value.value = right_squeeze
                with left_dex3_trigger_pressed.get_lock():
                    left_dex3_trigger_pressed.value = bool(tele_data.left_ctrl_trigger)
                with right_dex3_trigger_pressed.get_lock():
                    right_dex3_trigger_pressed.value = bool(tele_data.right_ctrl_trigger)
                with left_dex3_squeeze_pressed.get_lock():
                    left_dex3_squeeze_pressed.value = bool(tele_data.left_ctrl_squeeze)
                with right_dex3_squeeze_pressed.get_lock():
                    right_dex3_squeeze_pressed.value = bool(tele_data.right_ctrl_squeeze)
            elif args.ee == "dex1" and args.input_mode == "controller":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_ctrl_triggerValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_ctrl_triggerValue
            elif args.ee == "dex1" and args.input_mode == "hand":
                with left_gripper_value.get_lock():
                    left_gripper_value.value = tele_data.left_hand_pinchValue
                with right_gripper_value.get_lock():
                    right_gripper_value.value = tele_data.right_hand_pinchValue
            else:
                pass
            
            # high level control
            button_toggled_sync = False
            if args.input_mode == "controller":
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                    if args.motion:
                        loco_wrapper.Move(0.0, 0.0, 0.0)
                if not STOP:
                    START, left_sync_button_was_pressed, toggled = apply_sync_toggle(
                        START,
                        tele_data.left_ctrl_bButton,
                        left_sync_button_was_pressed,
                    )
                    if toggled:
                        button_toggled_sync = True
                        if START:
                            logger_mp.info("Arm sync resumed from left controller [Y/B].")
                        else:
                            held_arm_q = arm_ctrl.get_current_dual_arm_q().copy()
                            held_arm_tauff = held_arm_q * 0.0
                            logger_mp.info("Arm sync paused from left controller [Y/B]; walking remains active.")
                    if args.motion:
                        # command robot to enter damping mode. soft emergency stop function
                        if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                            loco_wrapper.Damp()
                        # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                        loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                          -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                          -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            if STOP:
                break

            if START and not sync_enabled_last:
                if not button_toggled_sync:
                    logger_mp.info("Arm sync enabled.")
            elif not START and sync_enabled_last:
                if not button_toggled_sync:
                    held_arm_q = arm_ctrl.get_current_dual_arm_q().copy()
                    held_arm_tauff = held_arm_q * 0.0
                    logger_mp.info("Arm sync paused; holding current arm pose.")
            sync_enabled_last = START

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            if START:
                # solve ik using motor data and wrist pose, then use ik results to control arms.
                time_ik_start = time.time()
                sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
                time_ik_end = time.time()
                logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")
                arm_ctrl.ctrl_dual_arm(sol_q, sol_tauff)
            else:
                sol_q = held_arm_q
                sol_tauff = held_arm_tauff
                arm_ctrl.ctrl_dual_arm(held_arm_q, held_arm_tauff)

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex3" and args.input_mode == "controller":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    states = {
                        "left_arm": {                                                                    
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],                          
                            "torque": [],                        
                        }, 
                        "right_arm": {                                                                    
                            "qpos":   right_arm_state.tolist(),       
                            "qvel":   [],                          
                            "torque": [],                         
                        },                        
                        "left_ee": {                                                                    
                            "qpos":   left_ee_state,           
                            "qvel":   [],                           
                            "torque": [],                          
                        }, 
                        "right_ee": {                                                                    
                            "qpos":   right_ee_state,       
                            "qvel":   [],                           
                            "torque": [],  
                        }, 
                        "body": {
                            "qpos": current_body_state,
                        }, 
                    }
                    actions = {
                        "left_arm": {                                   
                            "qpos":   left_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],      
                        }, 
                        "right_arm": {                                   
                            "qpos":   right_arm_action.tolist(),       
                            "qvel":   [],       
                            "torque": [],       
                        },                         
                        "left_ee": {                                   
                            "qpos":   left_hand_action,       
                            "qvel":   [],       
                            "torque": [],       
                        }, 
                        "right_ee": {                                   
                            "qpos":   right_hand_action,       
                            "qvel":   [],       
                            "torque": [], 
                        }, 
                        "body": {
                            "qpos": current_body_action,
                        }, 
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()            
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions, sim_state=sim_state)
                    else:
                        recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            if arm_ctrl is not None:
                arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        
        try:
            if args.ipc and ipc_server is not None:
                ipc_server.stop()
            else:
                if stop_listening_fn is not None:
                    stop_listening_fn()
                if listen_keyboard_thread is not None:
                    listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            if img_client is not None:
                img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            if tv_wrapper is not None:
                tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim and sim_state_subscriber is not None:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")
        
        try:
            if args.record and recorder is not None:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)
