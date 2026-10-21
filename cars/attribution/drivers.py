"""Braking controllers used to attribute responsibility in collision replays.

Each driver sees, at every 0.1 s step, the target's speed and the adversary's position and
velocity in the target's frame, and returns a braking deceleration (>= 0). All three are
longitudinal: the target keeps its path and only slows down.

Models and parameters:

FSM, fuzzy safety model (UN R157 Annex 4 Appendix 3; Mattas et al. 2020, 2022):
    reaction time tau = 0.75 s, comfortable / maximum braking of the target 4 / 6 m/s^2,
    maximum braking assumed for the adversary 7 m/s^2, safety margin d1 = 2 m, jerk 12.65 m/s^3.
    PFS (proactive): can the target stop behind the adversary if the adversary brakes hard?
        d_safe   = v tau + v^2 / (2 b_comf) - u^2 / (2 b_adv) + d1
        d_unsafe = v tau + v^2 / (2 b_max)  - u^2 / (2 b_adv)
        PFS = mu(gap - d1)
    CFS (critical): only when the target is faster than the adversary (v > u), with
        a' = max(a_target, -b_comf) and v* = v + a' tau. If v* <= u, d_safe = d_unsafe =
        (v - u)^2 / (2 |a'|). Otherwise d_new = ((v + v*)/2 - u) tau, then
        d_safe = d_new + (v* - u)^2 / (2 b_comf) and d_unsafe = d_new + (v* - u)^2 / (2 b_max).
        CFS = mu(gap)
    mu(x) = clip((x_safe - x) / (x_safe - x_unsafe), 0, 1), a step at x_safe if degenerate.
    Braking: CFS > 0 -> b_comf + CFS (b_max - b_comf); else PFS b_comf.
    Implementation choices: no braking when the adversary is not ahead (bumper gap <= 0). When
    the boxes are laterally apart, R157's cut-in pre-check applies: no braking unless the
    adversary moves toward the target, the target is faster by more than 0.1 m/s, and the
    lateral gap closes within the longitudinal closing time + 0.1 s. The first nonzero
    command starts a 0.75 s window without braking (counted down in 0.1 s steps); then
    braking ramps up at the jerk limit toward the current command, capped at 6 m/s^2.

CC-JP, careful and competent driver of the Japanese model (UN R157 Annex 4 Appendix 3),
lane-aware version:
    reacts once the adversary is ahead, closing faster than 0.1 m/s, with TTC < 2 s and
    a lateral body gap below dist_to_react = lane_width / 2 + (lane_width - adv_width) / 2
    - 0.72 - 0.375 (lane width from the nuScenes map at the target's start, 1.2 m if unknown).
    Then 0.75 s at 0.4 m/s^2 (releasing the accelerator), then a jerk-limited ramp to
    0.774 g. Once triggered, it brakes to the end.

RSS, responsibility-sensitive safety (Shalev-Shwartz et al. 2017):
    longitudinal safe distance with response time 0.75 s, target acceleration during the
    response 3 m/s^2, guaranteed braking 6 m/s^2, adversary braking 6 m/s^2; lateral safe
    distance with margin 0.3 m and adversary lateral acceleration 1 m/s^2. Reacts only
    when both are violated, then waits 0.75 s and ramps up to 0.774 g.
"""
import numpy as np

G = 9.81


class FSM:
    name = "FSM"

    def __init__(self, tau=0.75, b2_comf=4.0, b2_max=6.0, b1_max=7.0, d1=2.0, jerk=12.65,
                 max_decel_cap=6.0, dt=0.1):
        self.tau, self.b2_comf, self.b2_max, self.b1_max, self.d1 = tau, b2_comf, b2_max, b1_max, d1
        self.jerk, self.max_decel_cap, self.dt = jerk, max_decel_cap, dt
        self.reset()

    def reset(self):
        self.triggered, self.rt_counter, self.current_decel = False, 0.0, 0.0

    @staticmethod
    def mu(x, x_safe, x_unsafe):
        if x_safe <= x_unsafe:
            return 1.0 if x < x_safe else 0.0
        return float(np.clip((x_safe - x) / (x_safe - x_unsafe), 0.0, 1.0))

    def _ramp(self, target):
        if not self.triggered:
            if target > 0:
                self.triggered, self.rt_counter, self.current_decel = True, self.tau, 0.0
            return 0.0
        if self.rt_counter > 0:
            self.rt_counter -= self.dt
            return 0.0
        self.current_decel = min(self.current_decel + self.jerk * self.dt, target, self.max_decel_cap)
        return self.current_decel

    def __call__(self, s):
        d, u2, u1, a2 = s["distance"], s["target_speed"], s["adv_speed"], s["target_accel"]
        if s["d_lon_signed"] <= 0:                       # adversary not ahead
            s["_pfs"] = s["_cfs"] = 0.0
            return self._ramp(0.0)

        skip = False
        lat_gap = abs(s["lateral_offset"]) - s["half_width_target"] - s["half_width_adv"]
        if lat_gap > 0:                                  # R157 cut-in pre-check
            if s["adv_v_lat"] <= 0:
                skip = True
            else:
                closing = u2 - u1
                if closing <= 0.1:
                    skip = True
                else:
                    t_lat = lat_gap / s["adv_v_lat"]
                    t_lon = (s["d_lon_signed"] + 2 * s["half_length_target"] + 2 * s["half_length_adv"]) / closing
                    skip = t_lat > t_lon + 0.1
        if skip:
            s["_pfs"] = s["_cfs"] = 0.0
            return self._ramp(0.0)

        d_safe = u2 * self.tau + u2 ** 2 / (2 * self.b2_comf) - u1 ** 2 / (2 * self.b1_max) + self.d1
        d_unsafe = u2 * self.tau + u2 ** 2 / (2 * self.b2_max) - u1 ** 2 / (2 * self.b1_max)
        pfs = self.mu(d - self.d1, d_safe, d_unsafe)
        if u2 > u1:
            a2p = max(a2, -self.b2_comf)
            u2s = u2 + a2p * self.tau
            if u2s <= u1:
                denom = 2 * abs(a2p) if abs(a2p) > 1e-6 else 1e-6
                d_safe = d_unsafe = (u2 - u1) ** 2 / denom
            else:
                d_new = ((u2 + u2s) / 2 - u1) * self.tau
                d_safe = d_new + (u2s - u1) ** 2 / (2 * self.b2_comf)
                d_unsafe = d_new + (u2s - u1) ** 2 / (2 * self.b2_max)
            cfs = self.mu(d, d_safe, d_unsafe)
        else:
            cfs = 0.0
        s["_pfs"], s["_cfs"] = pfs, cfs
        target = cfs * (self.b2_max - self.b2_comf) + self.b2_comf if cfs > 0 else pfs * self.b2_comf
        return self._ramp(target)


class CCJP:
    name = "CC-JP-LaneAware"

    def __init__(self, rt=0.75, critical_ttc=2.0, release_decel=0.4, jerk=12.65,
                 max_decel=0.774 * G, dt=0.1, dist_to_react=1.2):
        self.rt, self.critical_ttc, self.release_decel = rt, critical_ttc, release_decel
        self.jerk, self.max_decel, self.dt = jerk, max_decel, dt
        self.dist_to_react = dist_to_react
        self.reset()

    def reset(self):
        self.triggered, self.rt_counter, self.current_decel = False, 0.0, 0.0

    def __call__(self, s):
        d, u2, u1 = s["distance"], s["target_speed"], s["adv_speed"]
        if s["d_lon_signed"] <= 0:
            return 0.0
        closing = u2 - u1
        if not self.triggered and closing > 0.1 and d > 0:
            lat_gap = abs(s["lateral_offset"]) - (s["half_width_target"] + s["half_width_adv"])
            if lat_gap < self.dist_to_react and d / closing < self.critical_ttc:
                self.triggered, self.rt_counter, self.current_decel = True, self.rt, self.release_decel
        if not self.triggered:
            return 0.0
        if self.rt_counter > 0:
            self.rt_counter -= self.dt
            self.current_decel = self.release_decel
            return self.release_decel
        self.current_decel = min(self.current_decel + self.jerk * self.dt, self.max_decel)
        return self.current_decel


class RSS:
    name = "RSS"

    def __init__(self, rt=0.75, max_accel=3.0, min_brake_guarantee=6.0, max_decel=0.774 * G,
                 adv_max_d=6.0, adv_max_a_lat=1.0, mu_lat=0.3, jerk=12.65, dt=0.1):
        self.rt, self.max_accel, self.min_brake_guarantee = rt, max_accel, min_brake_guarantee
        self.max_decel, self.adv_max_d, self.adv_max_a_lat, self.mu_lat = max_decel, adv_max_d, adv_max_a_lat, mu_lat
        self.jerk, self.dt = jerk, dt
        self.reset()

    def reset(self):
        self.triggered, self.rt_counter, self.current_decel = False, 0.0, 0.0

    def safe_lon(self, u2, u1):
        a = u2 * self.rt + self.max_accel * self.rt ** 2 / 2
        b = (u2 + self.rt * self.max_accel) ** 2 / (2 * self.min_brake_guarantee)
        c = u1 ** 2 / (2 * self.adv_max_d)
        return max(a + b - c, 0.0)

    def safe_lat(self, adv_v_lat):
        v = abs(adv_v_lat)
        return (self.mu_lat + abs((2 * v + self.adv_max_a_lat * self.rt) * self.rt / 2)
                + (v + self.adv_max_a_lat * self.rt) ** 2 / (2 * self.adv_max_a_lat))

    def __call__(self, s):
        d, u2, u1 = s["distance"], s["target_speed"], s["adv_speed"]
        if s["d_lon_signed"] <= 0:
            return 0.0
        if not self.triggered:
            lat_gap = abs(s["lateral_offset"]) - s["half_width_target"] - s["half_width_adv"]
            if d < self.safe_lon(u2, u1) and lat_gap < self.safe_lat(s["adv_v_lat"]):
                self.triggered, self.rt_counter, self.current_decel = True, self.rt, 0.0
        if not self.triggered:
            return 0.0
        if self.rt_counter > 0:
            self.rt_counter -= self.dt
            return 0.0
        self.current_decel = min(self.current_decel + self.jerk * self.dt, self.max_decel)
        return self.current_decel


def compute_dist_to_react(lane_width, adv_width, ego_dist_to_lane=None):
    """R157 Annex 4 Appendix 3; the target is assumed centred in its lane."""
    if ego_dist_to_lane is None:
        ego_dist_to_lane = lane_width / 2.0
    return ego_dist_to_lane + (lane_width - adv_width) / 2.0 - 0.72 - 0.375
