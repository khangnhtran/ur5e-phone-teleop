"""Pure pose adapter, no HEBI dependency; Python 3.8 compatible."""
import numpy as np
from scipy.spatial.transform import Rotation

def limited(v, maximum):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v * min(1., maximum / max(n, 1e-12))

class PhoneAdapter:
    def __init__(self, controller, pos_scale=0.5, rot_scale=0.5, translation_only=False,
                 allow_missing_quality=False, gripper_button=None, reset_button=None,
                 smooth_tau=.06, max_speed=.25, max_accel=1.5, orientation_lock=None, stale_seconds=.25, quality_stale_seconds=.5,
                 max_angular_speed=1.8, max_lag_m=.04, max_lag_rad=.30):
        self.c = controller
        self.max_angular_speed=max_angular_speed
        self.max_lag_m=max_lag_m
        self.max_lag_rad=max_lag_rad
        self.stale_seconds=stale_seconds
        self.quality_stale_seconds=quality_stale_seconds
        self.pos_scale, self.rot_scale = pos_scale, rot_scale
        self.translation_only = translation_only
        self.allow_missing_quality = allow_missing_quality
        self.gripper_button, self.reset_button = gripper_button, reset_button
        self.mapping = None
        self.session = None
        self.seq = -1
        self.last_fresh = None
        self._healthy = False
        self.armed = False
        self.active = False
        self.prev_buttons = {}
        self.closed = False
        self.grip_drive = 0.
        self.reset_requested = False
        self.status = "waiting_for_release"
        self.last_error = None
        self.target_p = np.array(controller.ee_pos, dtype=float)
        self.orientation_lock = None if orientation_lock is None else np.array(orientation_lock,dtype=float)
        self.target_r = np.array(controller.ee_ori_mat, dtype=float) if self.orientation_lock is None else self.orientation_lock.copy()
        self.center = self.target_p.copy()
        self.lower = self.center - np.array([.25, .25, .30])
        self.upper = self.center + np.array([.25, .25, .30])
        self.last_phone = None
        self.last_phone_time = None
        self.smooth_tau, self.max_speed, self.max_accel = smooth_tau, max_speed, max_accel
        self.command_p, self.command_r = self.target_p.copy(), self.target_r.copy()
        self.velocity = np.zeros(3)
        self.last_action_time = None

    def hold_command(self):
        # Stop advancing the trajectory when clutch/tracking is released.
        self.target_p, self.target_r = self.command_p.copy(), self.command_r.copy()
        self.velocity[:] = 0.
        if self.orientation_lock is not None:
            self.target_r=self.orientation_lock.copy()
            self.command_r=self.orientation_lock.copy()

    def stop(self, reason):
        self._healthy = False
        self.hold_command()
        self.active = self.armed = False
        self.grip_drive = 0.
        self.prev_buttons.clear()
        self.status = reason

    def ingest(self, m, now):
        try:
            if m.get("schema") != 1:
                raise ValueError("bad_schema")
            if m.get("session") != self.session:
                self.stop("receiver_restart")
                self.session, self.seq = m.get("session"), -1
            seq = int(m["seq"])
            if seq <= self.seq:
                return
            self.seq = seq
            transport_age=now-float(m["host_monotonic"])
            if not (0 <= transport_age <= self.stale_seconds):
                raise ValueError("stale_transport")
            if not all(0 <= float(m[k])+transport_age <= self.stale_seconds for k in ("packet_age_s","pose_age_s")):
                raise ValueError("stale_pose")
            p = np.asarray(m["position_m"], dtype=float)
            q = np.asarray(m["quaternion_wxyz"], dtype=float)
            if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
                raise ValueError("invalid_pose")
            if abs(np.linalg.norm(q)-1) > .05:
                raise ValueError("invalid_quaternion")
            # Normal is enum 6 in Python SDK; never substitute the MATLAB value.
            quality = m.get("ar_quality")
            if quality is None:
                if not self.allow_missing_quality:
                    raise ValueError("quality_unknown")
            elif quality != 6 or not (0 <= float(m["quality_age_s"])+transport_age <= self.quality_stale_seconds):
                raise ValueError("tracking_limited")
            buttons = m["buttons"]
            ages = m["button_age_s"]
            if buttons.get("B1") not in (0,1) or not (0 <= float(ages["B1"])+transport_age <= self.stale_seconds):
                raise ValueError("enable_unknown")
            r = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
            if buttons["B1"] and self.active and self.last_phone is not None:
                old_p, old_r = self.last_phone
                source_dt=0. if self.last_phone_time is None else max(0.,float(m["host_monotonic"])-self.last_phone_time)
                gap=min(source_dt,.15)
                if np.linalg.norm(p-old_p) > .15+gap or np.linalg.norm(Rotation.from_matrix(r @ old_r.T).as_rotvec()) > .8+3*gap:
                    raise ValueError("phone_pose_jump")
            self.last_error = None
            self.last_fresh = now
            self._healthy = True
            just_started=False
            if not buttons["B1"]:
                if self.active:
                    self.hold_command()
                self.active = False
                self.armed = True
                self.status = "clutch_released"
            elif self.armed:
                if not self.active:
                    just_started=True
                    if self.mapping is None:
                        # Calibration: screen up, phone top along robot world +X.
                        axes = np.array([[0,1,0],[-1,0,0],[0,0,1.]])
                        self.mapping = axes @ r.T
                    self.anchor_p, self.anchor_r = p.copy(), r.copy()
                    # Re-anchor to measured robot pose, avoiding residual old target.
                    self.anchor_robot_p = np.array(self.c.ee_pos)
                    self.anchor_robot_r = np.array(self.c.ee_ori_mat)
                    if hasattr(self.c,"sim"):
                        sid=self.c.sim.model.site_name2id(self.c.eef_name)
                        self.anchor_robot_p=np.array(self.c.sim.data.site_xpos[sid])
                        self.anchor_robot_r=np.array(self.c.sim.data.site_xmat[sid]).reshape(3,3)
                    self.target_p = self.anchor_robot_p.copy()
                    self.target_r = self.anchor_robot_r.copy() if self.orientation_lock is None else self.orientation_lock.copy()
                    self.command_p, self.command_r = self.target_p.copy(), self.target_r.copy()
                    self.velocity[:] = 0.
                    self.active = True
                self.target_p = np.clip(self.anchor_robot_p + self.pos_scale*self.mapping@(p-self.anchor_p),
                                        self.lower, self.upper)
                if not self.translation_only and self.orientation_lock is None and not just_started and self.last_phone is not None:
                    # Relative ARKit pose increments, not IMU integration. This also
                    # remains continuous across +/-pi and permits full wrist turns.
                    delta_r=self.mapping @ (r @ self.last_phone[1].T) @ self.mapping.T
                    rv=Rotation.from_matrix(delta_r).as_rotvec()*self.rot_scale
                    self.target_r=Rotation.from_rotvec(rv).as_matrix() @ self.target_r
                self.status = "moving"
            else:
                self.status = "release_B1_to_rearm"
            self.last_phone = (p.copy(), r.copy())
            self.last_phone_time=float(m["host_monotonic"])
            for pin, kind in ((self.gripper_button,"gripper"), (self.reset_button,"reset")):
                if pin is None:
                    continue
                value = buttons.get(pin)
                if value not in (0,1) or not (0 <= float(ages.get(pin, 1e9))+transport_age <= self.stale_seconds):
                    self.prev_buttons.pop(pin, None)
                    if kind == "gripper":
                        self.grip_drive = 0.
                    continue
                # Require observed release, including after reconnect or startup.
                if value == 1 and self.prev_buttons.get(pin) == 0:
                    if kind == "gripper":
                        self.closed = not self.closed
                        self.grip_drive = 1. if self.closed else -1.
                    else:
                        self.reset_requested = True
                self.prev_buttons[pin] = value
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self.last_error = str(exc)
            self.stop("invalid_or_stale_input")

    def is_healthy(self, now):
        return self._healthy and self.last_fresh is not None and 0 <= now-self.last_fresh <= self.stale_seconds

    def check_timeout(self, now):
        if self.last_fresh is None or now-self.last_fresh > self.stale_seconds:
            self.stop("phone_timeout")

    def action(self, now, control_dt=None):
        self.check_timeout(now)
        c = self.c
        dt = .05 if self.last_action_time is None else np.clip(now-self.last_action_time, 0., .1)
        if control_dt is not None:
            dt=float(control_dt)
        self.last_action_time = now
        if self.active and dt > 0:
            # Bound unconsumed input instead of accumulating a distant goal.
            # Rebase the clutch offset so a fast move does not leave a long tail.
            bounded=np.asarray(c.ee_pos)+limited(self.target_p-c.ee_pos,self.max_lag_m)
            self.anchor_robot_p+=bounded-self.target_p
            self.target_p=bounded
            self.command_p=np.asarray(c.ee_pos)+limited(self.command_p-c.ee_pos,self.max_lag_m)
            if self.orientation_lock is None and not self.translation_only:
                error_r=Rotation.from_matrix(self.target_r@c.ee_ori_mat.T).as_rotvec()
                self.target_r=Rotation.from_rotvec(limited(error_r,self.max_lag_rad)).as_matrix()@c.ee_ori_mat
                command_r=Rotation.from_matrix(self.command_r@c.ee_ori_mat.T).as_rotvec()
                self.command_r=Rotation.from_rotvec(limited(command_r,self.max_lag_rad)).as_matrix()@c.ee_ori_mat
            error = self.target_p - self.command_p
            desired_v = limited(error / max(self.smooth_tau, dt), self.max_speed)
            self.velocity += limited(desired_v-self.velocity, self.max_accel*dt)
            step = self.velocity*dt
            if np.linalg.norm(step) > np.linalg.norm(error) and np.dot(step,error) > 0:
                step = error
                self.velocity = step/dt
            self.command_p += step
            rv = Rotation.from_matrix(self.target_r @ self.command_r.T).as_rotvec()
            alpha = 1.-np.exp(-dt/max(self.smooth_tau, 1e-6))
            rot_step = limited(alpha*rv, self.max_angular_speed*dt)
            self.command_r = Rotation.from_rotvec(rot_step).as_matrix() @ self.command_r
        # Controller error envelope is separate from commanded trajectory speed.
        dp = limited(self.command_p - c.ee_pos, .04)
        dr = limited(Rotation.from_matrix(self.command_r @ c.ee_ori_mat.T).as_rotvec(), .25)
        physical = np.r_[dp, dr]
        out_min, out_max = np.asarray(c.output_min), np.asarray(c.output_max)
        in_min, in_max = np.asarray(c.input_min), np.asarray(c.input_max)
        # Invert Controller.scale_action; OSC uses R_delta @ R_current.
        arm = (physical - (out_max+out_min)/2) / ((out_max-out_min)/(in_max-in_min)) + (in_max+in_min)/2
        return np.r_[np.clip(arm, in_min, in_max), self.grip_drive]
