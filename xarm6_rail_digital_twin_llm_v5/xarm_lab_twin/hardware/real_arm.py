# hardware/real_arm.py
"""
Real xArm6 hardware wrapper. Only imported when --mode real is used.
Requires `pip install xarm-python-sdk` and a network-accessible xArm controller.
"""
from xarm.wrapper import XArmAPI


class RealXArmAPI:
    def __init__(self, ip: str):
        self.arm = XArmAPI(ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self._rail_pos_mm = 0.0

    def set_rail_position(self, position_mm: float,
                          speed_mm_s: float = 50.0,
                          wait: bool = True, **kwargs) -> int:
        try:
            ret = self.arm.set_linear_track_pos(
                position_mm, speed=speed_mm_s, wait=wait)
            self._rail_pos_mm = position_mm
            return ret if isinstance(ret, int) else 0
        except AttributeError:
            print(f"[RealArm] set_linear_track_pos unavailable - "
                  f"check firmware/SDK version")
            self._rail_pos_mm = position_mm
            return 0

    def get_rail_position(self) -> tuple:
        try:
            ret = self.arm.get_linear_track_pos()
            if isinstance(ret, tuple):
                return ret
            return 0, float(ret)
        except AttributeError:
            return 0, self._rail_pos_mm

    def set_position(self, x, y, z, roll=0, pitch=0, yaw=0,
                     speed=100, wait=True, **kwargs) -> int:
        return self.arm.set_position(
            x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw,
            speed=speed, wait=wait
        )

    def set_servo_angle(self, angle, speed=30, wait=True, **kwargs) -> int:
        return self.arm.set_servo_angle(angle=angle, speed=speed, wait=wait)

    def get_position(self):     return self.arm.get_position()
    def get_servo_angle(self):  return self.arm.get_servo_angle()
    def open_lite6_gripper(self):  return self.arm.open_lite6_gripper()
    def close_lite6_gripper(self): return self.arm.close_lite6_gripper()
    def motion_enable(self, enable=True): return self.arm.motion_enable(enable=enable)
    def set_mode(self, mode):             return self.arm.set_mode(mode)
    def set_state(self, state):           return self.arm.set_state(state)

    def disconnect(self):
        self.arm.disconnect()
