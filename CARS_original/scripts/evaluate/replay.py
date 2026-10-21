#!/usr/bin/env python3
"""Re-evaluate adversarial scenarios with three CCDMs (FSM / CC-JP-LaneAware / RSS).

This is the single, self-contained canonical paper script. It does NOT import
from a separate replay_baselines.py: every helper, controller, and replay
routine is defined in this file.

FSM implementation follows Mattas et al. 2022 (Accident Analysis & Prevention
174, 106743), §3.4 Eqs. 5-16 and Table 2. The JRC reference code
(github.com/ec-jrc/JRC-FSM) was consulted but is NOT authoritative; where the
two disagree, the paper wins.

The 7 bugs from the pre-audit version of replay_baselines.py are fixed:
    1. Reaction force is paper Eq. 16 piecewise (not max()).
    2. PFS uses both `+d_1` in d_safe and `(d-d_1)` as membership input.
    3. FSM is now stateful with τ-delay + jerk-limited ramp-up.
    4. b_1max default is 7 m/s² (paper Table 2), not 8.
    5. Lateral safety pre-check (paper Eq. 5) gates the fuzzy stage.
    6. CFS short-circuits to 0 when u_2 ≤ u_1 (no false positives).
    7. target_accel is signed instantaneous deceleration, not a positive
       running average.

CC-JP-LaneAware follows UN R157 Annex 4 Appendix 3:
    dist_to_react = ego_dist_to_lane + (lane_width - adv_width)/2 - 0.72 - 0.375
    trigger iff lateral_body_gap < dist_to_react AND TTC < 2 s

Falls back to dist_to_react = 1.20 m when nuScenes map lookup fails (non-nuScenes
HDF5 or missing lane information).

RSS follows Shalev-Shwartz et al. 2017 (longitudinal-only).

Usage:
  python -m scripts.evaluate replay \
      --hdf5_path <run_dir>/data.hdf5 \
      --output_dir <run_dir>/replay_results/
"""

import argparse
import csv
import os
import re
import time
from abc import ABC, abstractmethod

import h5py
import numpy as np
from tqdm import tqdm


# =============================================================================
# Geometry helpers
# =============================================================================

def _batch_rotate_2d(xy, theta):
    x1 = xy[..., 0] * np.cos(theta) - xy[..., 1] * np.sin(theta)
    y1 = xy[..., 1] * np.cos(theta) + xy[..., 0] * np.sin(theta)
    return np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1)), axis=-1)


def obb_collision(pos1, yaw1, ext1, pos2, yaw2, ext2,
                  offsetX=1.0, offsetY=0.3):
    """OBB collision check at one timestep, matching VEH_VEH_collision (rollout)."""
    p1 = np.array([pos1[0], pos1[1], yaw1]).reshape(1, 3)
    p2 = np.array([pos2[0], pos2[1], yaw2]).reshape(1, 3)
    S1 = np.array([[ext1[0] + offsetX, ext1[1] + offsetY]], dtype=np.float64)
    S2 = np.array([[ext2[0] + offsetX, ext2[1] + offsetY]], dtype=np.float64)

    cornersX = np.kron(S1[..., 0], np.array([0.5, 0.5, -0.5, -0.5]))
    cornersY = np.kron(S1[..., 1], np.array([0.5, -0.5, 0.5, -0.5]))
    corners = np.stack([cornersX, cornersY], axis=-1)
    theta1 = p1[..., 2]
    theta2 = p2[..., 2]
    dx = (p1[..., 0:2] - p2[..., 0:2]).repeat(4, axis=-2)
    delta_x1 = _batch_rotate_2d(corners, theta1.repeat(4, axis=-1)) + dx
    delta_x2 = _batch_rotate_2d(delta_x1, -theta2.repeat(4, axis=-1))
    dis = np.maximum(
        np.abs(delta_x2[..., 0]) - 0.5 * S2[..., 0].repeat(4, axis=-1),
        np.abs(delta_x2[..., 1]) - 0.5 * S2[..., 1].repeat(4, axis=-1),
    ).reshape(*S1.shape[:-1], 4)
    min_dis = float(np.min(dis, axis=-1)[0])
    return min_dis < 0, min_dis


def longitudinal_projection(vec, yaw):
    """Project a 2D vector onto the heading direction."""
    return vec[0] * np.cos(yaw) + vec[1] * np.sin(yaw)


def lateral_projection(vec, yaw):
    """Project a 2D vector onto the lateral (left-positive) direction."""
    return -vec[0] * np.sin(yaw) + vec[1] * np.cos(yaw)


def _build_arc_length(pos):
    diffs = np.diff(pos, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _interp_angle(arc_s, orig_arc, orig_yaw):
    idx = np.searchsorted(orig_arc, arc_s, side='right') - 1
    idx = int(np.clip(idx, 0, len(orig_arc) - 2))
    t0, t1 = orig_arc[idx], orig_arc[idx + 1]
    if t1 == t0:
        return float(orig_yaw[idx])
    alpha = (arc_s - t0) / (t1 - t0)
    y0, y1 = float(orig_yaw[idx]), float(orig_yaw[idx + 1])
    diff = (y1 - y0 + np.pi) % (2 * np.pi) - np.pi
    return y0 + alpha * diff


def _arc_to_world(arc_s, orig_arc, orig_pos, orig_yaw):
    arc_s = float(np.clip(arc_s, orig_arc[0], orig_arc[-1]))
    pos = np.array([
        np.interp(arc_s, orig_arc, orig_pos[:, 0]),
        np.interp(arc_s, orig_arc, orig_pos[:, 1]),
    ])
    yaw = _interp_angle(arc_s, orig_arc, orig_yaw)
    return pos, yaw


# =============================================================================
# Controllers
# =============================================================================

class BaseController(ABC):
    """Base class for longitudinal braking controllers."""

    handles_own_reaction: bool = True  # all CCDMs in this file are stateful

    @abstractmethod
    def compute_deceleration(self, state: dict) -> float:
        """Return deceleration magnitude (>=0). 0 means no braking."""

    def reset(self) -> None:
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class FSMController(BaseController):
    """Fuzzy Safety Model — Mattas et al. 2022 §3.4, Table 2.

    Stateful. Internally manages the τ reaction window and the jerk-limited
    ramp-up to the target deceleration.
    """

    handles_own_reaction = True

    def __init__(self, tau=0.75, b2_comf=4.0, b2_max=6.0, b1_max=7.0,
                 d1=2.0, jerk=12.65, max_decel_cap=6.0, dt=0.1):
        self.tau = tau
        self.b2_comf = b2_comf
        self.b2_max = b2_max
        self.b1_max = b1_max
        self.d1 = d1
        self.jerk = jerk
        self.max_decel_cap = max_decel_cap
        self.dt = dt
        self.reset()

    @property
    def name(self):
        return "FSM"

    def reset(self) -> None:
        self.triggered = False
        self.rt_counter = 0.0
        self.current_decel = 0.0

    @staticmethod
    def _fuzzy_mu(x, x_safe, x_unsafe):
        """μ = clamp((x_safe − x) / (x_safe − x_unsafe), 0, 1).
        Degenerate (x_safe ≤ x_unsafe) → step function: 1 if x < x_safe else 0."""
        if x_safe <= x_unsafe:
            return 1.0 if x < x_safe else 0.0
        return float(np.clip((x_safe - x) / (x_safe - x_unsafe), 0.0, 1.0))

    def compute_deceleration(self, state: dict) -> float:
        d  = state["distance"]
        u2 = state["target_speed"]
        u1 = state["adv_speed"]
        a2 = state.get("target_accel", 0.0)
        d_lon_signed = state.get("d_lon_signed", d)  # signed; falls back to d (>=0)

        # Gate 0 — paper §3.4 lateral criterion 1 ("cut-in is downstream"):
        # if adversary is at or behind ego, no rear-end risk for ego.
        if d_lon_signed <= 0:
            state["_pfs"] = 0.0
            state["_cfs"] = 0.0
            target_decel = 0.0
            return self._advance_state_machine(target_decel)

        # Gate 1 — lateral pre-check (UN R157 §3.4.2.1, page 57). Lateral risk
        # is identified ONLY if all four conditions hold:
        #   (a) adv ahead of ego longitudinally     — handled by Gate 0 above
        #   (b) adv moving toward ego laterally     — checked here
        #   (c) u_ego > u_cut_in                    — checked here (★)
        #   (d) T_lat < T_lon + 0.1                 — checked here
        # If any of (b)(c)(d) fails, no lateral risk → skip PFS/CFS.
        d_lat           = state.get("lateral_offset", 0.0)
        half_w_target   = state.get("half_width_target", 0.95)
        half_w_adv      = state.get("half_width_adv", 0.95)
        half_len_target = state.get("half_length_target", 2.5)
        half_len_adv    = state.get("half_length_adv", 2.5)
        adv_v_lat       = state.get("adv_v_lat", 0.0)  # toward target = positive

        skip_fuzzy = False
        lat_body_gap = abs(d_lat) - half_w_target - half_w_adv
        if lat_body_gap > 0:
            if adv_v_lat <= 0:
                # (b) adv moving away laterally → no risk
                skip_fuzzy = True
            else:
                closing = u2 - u1
                if closing <= 0.1:
                    # (c) ego not faster than adv → no rear-end-style risk per §3.4.2.1
                    skip_fuzzy = True
                else:
                    T_lat = lat_body_gap / adv_v_lat
                    # T_lon per UN R157 §3.4.2.1 (d):
                    #   numerator = distlon + length_ego + length_cut_in
                    # With distlon = bumper-to-bumper (d_gap = d_lon_signed
                    # after the §3.4.2.1 (a) bumper convention) and lengths =
                    # full vehicle lengths (= 2 × half_lengths), the numerator
                    # is d_gap + 2·half_len_target + 2·half_len_adv. This
                    # equals the JRC convention `|d_lon_centroid| + half_lens`
                    # but tracks the bumper interpretation directly.
                    T_lon = (d_lon_signed + 2 * half_len_target + 2 * half_len_adv) / closing
                    if T_lat > T_lon + 0.1:
                        # (d) time-race lost → no risk
                        skip_fuzzy = True

        if skip_fuzzy:
            state["_pfs"] = 0.0
            state["_cfs"] = 0.0
            target_decel = 0.0
        else:
            # PFS — paper Eq. 7-9 with d_1 margin
            d_safe_pfs   = u2*self.tau + u2**2/(2*self.b2_comf) - u1**2/(2*self.b1_max) + self.d1
            d_unsafe_pfs = u2*self.tau + u2**2/(2*self.b2_max)  - u1**2/(2*self.b1_max)
            pfs = self._fuzzy_mu(d - self.d1, d_safe_pfs, d_unsafe_pfs)

            # CFS — paper Eq. 10-15, only when ego closes on lead
            if u2 > u1:
                a2_prime = max(a2, -self.b2_comf)
                u2_star = u2 + a2_prime * self.tau
                if u2_star <= u1:
                    denom = 2 * abs(a2_prime) if abs(a2_prime) > 1e-6 else 1e-6
                    d_safe_cfs = d_unsafe_cfs = (u2 - u1)**2 / denom
                else:
                    d_new = ((u2 + u2_star) / 2 - u1) * self.tau
                    d_safe_cfs   = d_new + (u2_star - u1)**2 / (2*self.b2_comf)
                    d_unsafe_cfs = d_new + (u2_star - u1)**2 / (2*self.b2_max)
                cfs = self._fuzzy_mu(d, d_safe_cfs, d_unsafe_cfs)
            else:
                cfs = 0.0

            state["_pfs"] = pfs
            state["_cfs"] = cfs

            # Reaction force — paper Eq. 16 (piecewise)
            if cfs > 0:
                target_decel = cfs * (self.b2_max - self.b2_comf) + self.b2_comf
            else:
                target_decel = pfs * self.b2_comf

        return self._advance_state_machine(target_decel)

    def _advance_state_machine(self, target_decel: float) -> float:
        """τ reaction window, then jerk-limited ramp toward `target_decel`."""
        if not self.triggered:
            if target_decel > 0:
                self.triggered = True
                self.rt_counter = self.tau
                self.current_decel = 0.0
            return 0.0

        if self.rt_counter > 0:
            self.rt_counter -= self.dt
            return 0.0

        self.current_decel = min(
            self.current_decel + self.jerk * self.dt,
            target_decel,
            self.max_decel_cap,
        )
        return self.current_decel


class CCHumanDriverController(BaseController):
    """Competent and Careful Human Driver — UN R157 Annex 4 Appendix 3 (JP).

    Stateful: reaction-time release-pedal phase, then jerk ramp.
    Implementation matches JRC-FSM `CC_react` / `CC_check_safety_lane_info`.
    """

    handles_own_reaction = True

    def __init__(self, rt=0.75, critical_ttc=2.0, release_decel=0.4,
                 jerk=12.65, max_decel=0.774 * 9.81, dt=0.1):
        self.rt = rt
        self.critical_ttc = critical_ttc
        self.release_decel = release_decel
        self.jerk = jerk
        self.max_decel = max_decel
        self.dt = dt
        self.reset()

    @property
    def name(self):
        return "CC-JP"

    def reset(self) -> None:
        self.triggered = False
        self.rt_counter = 0.0
        self.current_decel = 0.0

    def compute_deceleration(self, state: dict) -> float:
        d  = state["distance"]
        u2 = state["target_speed"]
        u1 = state["adv_speed"]

        closing = u2 - u1
        if not self.triggered:
            if closing > 0.1 and d > 0:
                ttc = d / closing
                if ttc < self.critical_ttc:
                    self.triggered = True
                    self.rt_counter = self.rt
                    self.current_decel = self.release_decel

        if not self.triggered:
            return 0.0

        if self.rt_counter > 0:
            self.rt_counter -= self.dt
            self.current_decel = self.release_decel
            return self.release_decel

        self.current_decel = min(
            self.current_decel + self.jerk * self.dt, self.max_decel
        )
        return self.current_decel


class CCJPLaneAwareController(CCHumanDriverController):
    """CC-JP with R157 lane-width-aware lateral trigger.

    Trigger condition (`CC_check_safety_lane_info`):
        lateral_body_gap < dist_to_react  AND  TTC < critical_ttc

    `dist_to_react` is computed per-episode from the actual nuScenes lane
    width via `compute_dist_to_react`; `set_dist_to_react` updates it.
    """

    def __init__(self, rt=0.75, critical_ttc=2.0, release_decel=0.4,
                 jerk=12.65, max_decel=0.774 * 9.81, dt=0.1):
        super().__init__(rt=rt, critical_ttc=critical_ttc,
                         release_decel=release_decel, jerk=jerk,
                         max_decel=max_decel, dt=dt)
        self.dist_to_react = 1.2  # default before per-episode override

    @property
    def name(self):
        return "CC-JP-LaneAware"

    def set_dist_to_react(self, value: float) -> None:
        self.dist_to_react = float(value)

    def compute_deceleration(self, state: dict) -> float:
        d  = state["distance"]
        u2 = state["target_speed"]
        u1 = state["adv_speed"]
        lat_offset = state.get("lateral_offset", 0.0)
        hw_t = state.get("half_width_target", 0.95)
        hw_a = state.get("half_width_adv", 0.95)
        d_lon_signed = state.get("d_lon_signed", d)

        # Downstream check: if adv is at/behind ego, no rear-end risk.
        if d_lon_signed <= 0:
            return 0.0

        closing = u2 - u1

        if not self.triggered:
            if closing > 0.1 and d > 0:
                lat_body_gap = abs(lat_offset) - (hw_t + hw_a)
                ttc = d / closing
                if lat_body_gap < self.dist_to_react and ttc < self.critical_ttc:
                    self.triggered = True
                    self.rt_counter = self.rt
                    self.current_decel = self.release_decel

        if not self.triggered:
            return 0.0

        if self.rt_counter > 0:
            self.rt_counter -= self.dt
            self.current_decel = self.release_decel
            return self.release_decel

        self.current_decel = min(
            self.current_decel + self.jerk * self.dt, self.max_decel
        )
        return self.current_decel


class RSSController(BaseController):
    """Responsibility-Sensitive Safety — Shalev-Shwartz et al. 2017.

    Implementation matches JRC-FSM `RSS_check_safety` + `RSS_react` exactly:
    - Two distinct decel parameters (paper RSS terminology):
        min_brake_guarantee : the deceleration ego is GUARANTEED to be able
                              to apply, used in d_safe formula (JRC max_d=6).
        max_decel           : the actual ramp cap during reaction
                              (JRC RSS_max_deceleration = 0.774·g ≈ 7.59).
      Earlier versions of this script merged both into a single max_decel,
      which made d_safe artificially small and delayed triggering.
    - Trigger requires BOTH longitudinal AND lateral RSS violations
      (matches JRC models.py:107-124). Lateral-only or longitudinal-only
      violations are SAFE per RSS.

    d_safe_lon = u2·rt + max_a·rt²/2 + (u2 + rt·max_a)² / (2·min_brake_guarantee)
                                       - u1² / (2·adv_max_d)
    d_safe_lat = mu_lat
                + |(2·v_lat + adv_max_a_lat·rt)·rt / 2|
                + (v_lat + adv_max_a_lat·rt)² / (2·adv_max_a_lat)

    Stateful: rt window of coasting (no braking), then jerk ramp.
    """

    handles_own_reaction = True

    def __init__(self, rt=0.75, max_accel=3.0,
                 min_brake_guarantee=6.0,        # JRC vehicle.py:38 (max_d)
                 max_decel=0.774 * 9.81,          # JRC vehicle.py:65
                 adv_max_d=6.0,
                 adv_max_a_lat=1.0,               # JRC vehicle.py:39 (max_a_lat)
                 mu_lat=0.3,                      # JRC vehicle.py:66 (RSS_mu)
                 jerk=12.65, dt=0.1):
        self.rt = rt
        self.max_accel = max_accel
        self.min_brake_guarantee = min_brake_guarantee
        self.max_decel = max_decel
        self.adv_max_d = adv_max_d
        self.adv_max_a_lat = adv_max_a_lat
        self.mu_lat = mu_lat
        self.jerk = jerk
        self.dt = dt
        self.reset()

    @property
    def name(self):
        return "RSS"

    def reset(self) -> None:
        self.triggered = False
        self.rt_counter = 0.0
        self.current_decel = 0.0

    def _rss_safe_distance_lon(self, u2, u1):
        """Longitudinal RSS safe distance (JRC max_d = 6 in denominator)."""
        part_a = u2 * self.rt + self.max_accel * self.rt**2 / 2
        part_b = (u2 + self.rt * self.max_accel)**2 / (2 * self.min_brake_guarantee)
        part_c = u1**2 / (2 * self.adv_max_d)
        return max(part_a + part_b - part_c, 0.0)

    def _rss_safe_distance_lat(self, adv_v_lat):
        """Lateral RSS safe distance (JRC models.py:117-120)."""
        cut_lat = abs(adv_v_lat)
        return (
            self.mu_lat
            + abs((2 * cut_lat + self.adv_max_a_lat * self.rt) * self.rt / 2)
            + (cut_lat + self.adv_max_a_lat * self.rt)**2 / (2 * self.adv_max_a_lat)
        )

    def compute_deceleration(self, state: dict) -> float:
        d  = state["distance"]
        u2 = state["target_speed"]
        u1 = state["adv_speed"]
        d_lat        = state.get("lateral_offset", 0.0)
        half_w_t     = state.get("half_width_target", 0.95)
        half_w_a     = state.get("half_width_adv", 0.95)
        adv_v_lat    = state.get("adv_v_lat", 0.0)
        d_lon_signed = state.get("d_lon_signed", d)

        # Downstream check: if adv is at/behind ego, no rear-end risk.
        if d_lon_signed <= 0:
            return 0.0

        if not self.triggered:
            d_safe_lon   = self._rss_safe_distance_lon(u2, u1)
            d_safe_lat   = self._rss_safe_distance_lat(adv_v_lat)
            lat_body_gap = abs(d_lat) - half_w_t - half_w_a

            # Trigger only if BOTH longitudinal AND lateral are violated.
            if d < d_safe_lon and lat_body_gap < d_safe_lat:
                self.triggered = True
                self.rt_counter = self.rt
                self.current_decel = 0.0

        if not self.triggered:
            return 0.0

        if self.rt_counter > 0:
            self.rt_counter -= self.dt
            return 0.0

        self.current_decel = min(
            self.current_decel + self.jerk * self.dt, self.max_decel
        )
        return self.current_decel


# =============================================================================
# Replay engine
# =============================================================================

def check_raw_collision(target_pos, target_yaw, adv_pos, adv_yaw,
                        target_ext, adv_ext):
    """Replay raw HDF5 centroids through OBB collision (no controller).
    Used to verify the recorded trajectory itself contains a collision."""
    T = min(len(target_pos), len(adv_pos))
    min_dist = np.inf
    collision_step = None

    for t in range(T):
        hit, obb_dist = obb_collision(target_pos[t], target_yaw[t], target_ext,
                                      adv_pos[t], adv_yaw[t], adv_ext,
                                      offsetX=1.0, offsetY=0.3)
        if obb_dist < min_dist:
            min_dist = obb_dist
        if hit:
            collision_step = t
            break

    return collision_step is not None, collision_step, min_dist


def replay_episode(target_pos, target_yaw, adv_pos, adv_yaw,
                   target_ext, adv_ext, controller, dt=0.1):
    """Roller-coaster re-simulation. Target follows its original arc-length
    path; controller decides speed only. Adversary trajectory is fixed.

    All controllers in this script handle their own reaction window
    (handles_own_reaction = True), so there is NO outer reaction-time delay.

    target_accel passed to controllers is signed instantaneous deceleration:
        a₂_signed = -(speed_deficit_t - speed_deficit_{t-1}) / dt
    Negative means actively decelerating, matching paper Eq. 13's expectation.
    """
    T = min(len(target_pos), len(adv_pos))
    if T < 2:
        return {"preventable": False, "collision_step": None,
                "min_distance": np.inf, "max_pfs": 0.0, "max_cfs": 0.0,
                "criticality": "Easy", "speed_at_collision": np.nan,
                "collision_angle": np.nan, "min_ttc": np.inf,
                "max_braking_deficit": np.nan, "unpreventable_reason": None,
                "timeseries": {}}

    controller.reset()

    orig_arc = _build_arc_length(target_pos)
    orig_speed = np.diff(orig_arc) / dt

    speed_deficit = 0.0
    speed_deficit_prev = 0.0
    arc_s = 0.0

    half_len_target = float(target_ext[0]) / 2.0
    half_len_adv    = float(adv_ext[0]) / 2.0
    half_wid_target = float(target_ext[1]) / 2.0
    half_wid_adv    = float(adv_ext[1]) / 2.0

    timeseries = {
        "pfs": [], "cfs": [], "a_ctrl": [], "v_target": [], "distance": [],
        "ttc": [], "closing_speed": [], "required_braking_dist": [],
        "braking_deficit": [],
    }
    collision_step = None
    min_distance = np.inf
    trigger_step = None  # first step where the controller transitioned to triggered=True

    for t in range(T - 1):
        sim_pos, sim_yaw = _arc_to_world(arc_s, orig_arc, target_pos, target_yaw)

        gap_vec = adv_pos[t] - sim_pos
        d_lon = longitudinal_projection(gap_vec, sim_yaw)
        d_lat = lateral_projection(gap_vec, sim_yaw)
        d_gap = d_lon - (half_len_target + half_len_adv)
        # Gate 0 uses bumper-to-bumper signed distance per UN R157 §3.4.2.1 (a):
        # "The rear of the 'other vehicle' is ahead of the front of the ALKS
        #  vehicle along the longitudinal direction of motion".
        # i.e. (a) holds iff d_gap > 0 (bumpers separated, adv ahead).
        # JRC line 192 uses centroid (ego.pos_long > cut.pos_long), which is a
        # simplification that proceeds to PFS/CFS during the bumper-overlap-
        # but-centroids-apart window. Per regulation > paper > JRC, follow (a)
        # strictly. Bumper-overlap frames are out of FSM's modelled domain.
        d_lon_signed = d_gap  # signed bumper-to-bumper distance

        # Adversary kinematics in target's frame
        if t < T - 2:
            adv_vel = (adv_pos[t + 1] - adv_pos[t]) / dt
        else:
            adv_vel = (adv_pos[t] - adv_pos[t - 1]) / dt
        u1 = longitudinal_projection(adv_vel, sim_yaw)
        # adv lateral velocity, positive when moving toward target
        adv_lat_vel_signed = lateral_projection(adv_vel, sim_yaw)
        adv_v_lat_toward = -np.sign(d_lat) * adv_lat_vel_signed if d_lat != 0 else 0.0

        v_orig_t = float(orig_speed[t])
        v_target = max(0.0, v_orig_t - speed_deficit)

        # Signed instantaneous deceleration of target (paper Eq. 13 expectation)
        a2_signed = -(speed_deficit - speed_deficit_prev) / dt

        state = {
            "distance":           max(d_gap, 0.0),
            "d_lon_signed":       d_lon_signed,    # bumper-to-bumper signed; UN R157 §3.4.2.1 (a)
            "target_speed":       v_target,
            "adv_speed":          u1,
            "target_accel":       a2_signed,
            "lateral_offset":     d_lat,
            "half_width_target":  half_wid_target,
            "half_width_adv":     half_wid_adv,
            "half_length_target": half_len_target,
            "half_length_adv":    half_len_adv,
            "adv_v_lat":          adv_v_lat_toward,
        }
        a_ctrl = controller.compute_deceleration(state)
        # Detect first step where controller transitioned to triggered (FSM/CC-JP/RSS)
        if trigger_step is None and getattr(controller, "triggered", False):
            trigger_step = t

        # Derived metrics
        closing = v_target - u1
        ttc = (d_gap / closing) if (closing > 0.1 and d_gap > 0) else np.inf
        req_braking = v_target ** 2 / (2 * 6.0)
        b_deficit = req_braking - max(d_gap, 0.0)

        timeseries["a_ctrl"].append(a_ctrl)
        timeseries["v_target"].append(v_target)
        timeseries["distance"].append(d_gap)
        timeseries["pfs"].append(state.get("_pfs", 0.0))
        timeseries["cfs"].append(state.get("_cfs", 0.0))
        timeseries["ttc"].append(ttc)
        timeseries["closing_speed"].append(closing)
        timeseries["required_braking_dist"].append(req_braking)
        timeseries["braking_deficit"].append(b_deficit)

        # Accumulate braking. Stateful controllers gate themselves.
        speed_deficit_prev = speed_deficit
        speed_deficit += a_ctrl * dt

        v_target = max(0.0, v_orig_t - speed_deficit)
        arc_s += v_target * dt

        next_pos, next_yaw = _arc_to_world(arc_s, orig_arc, target_pos, target_yaw)
        hit, obb_dist = obb_collision(next_pos, next_yaw, target_ext,
                                      adv_pos[t + 1], adv_yaw[t + 1], adv_ext)
        min_distance = min(min_distance, obb_dist)
        if hit:
            collision_step = t + 1
            break

    preventable = collision_step is None
    ts = {k: np.array(v) for k, v in timeseries.items()}

    max_pfs = float(np.max(ts["pfs"])) if len(ts["pfs"]) > 0 else 0.0
    max_cfs = float(np.max(ts["cfs"])) if len(ts["cfs"]) > 0 else 0.0
    if max_cfs >= 0.9:
        criticality = "Hard"
    elif max_pfs > 0.85:
        criticality = "Medium"
    else:
        criticality = "Easy"

    col_idx = collision_step - 1 if (collision_step is not None and collision_step > 0) else None

    speed_at_collision = float(ts["v_target"][col_idx]) if col_idx is not None else np.nan
    if col_idx is not None and col_idx < len(target_yaw) and col_idx < len(adv_yaw):
        raw_angle = float(adv_yaw[col_idx]) - float(target_yaw[col_idx])
        collision_angle = float((raw_angle + np.pi) % (2 * np.pi) - np.pi)
    else:
        collision_angle = np.nan

    finite_ttc = ts["ttc"][np.isfinite(ts["ttc"])]
    min_ttc = float(np.min(finite_ttc)) if len(finite_ttc) > 0 else np.inf
    max_braking_deficit = float(np.max(ts["braking_deficit"])) if len(ts["braking_deficit"]) > 0 else np.nan
    # BD value at the moment the controller first transitions to triggered=True.
    # Aligns with paper's "BD+ at the reaction boundary" claim (was previously
    # reported as np.max over the whole episode).
    if trigger_step is not None and trigger_step < len(ts["braking_deficit"]):
        bd_at_trigger = float(ts["braking_deficit"][trigger_step])
    else:
        bd_at_trigger = np.nan

    unpreventable_reason = None
    if not preventable:
        reasons = []
        if max_cfs >= 0.9:
            reasons.append(f"CFS={max_cfs:.2f}>=0.9 (emergency zone)")
        elif max_pfs >= 0.85:
            reasons.append(f"PFS={max_pfs:.2f}>=0.85 (comfort zone)")
        if max_braking_deficit > 0:
            reasons.append(f"max_braking_deficit={max_braking_deficit:.2f}m>0")
        if np.isfinite(min_ttc) and min_ttc < 1.0:
            reasons.append(f"min_TTC={min_ttc:.2f}s<1.0s")
        if not np.isnan(speed_at_collision) and speed_at_collision > 5.0:
            reasons.append(f"v_col={speed_at_collision:.1f}m/s")
        unpreventable_reason = "; ".join(reasons) if reasons else "late approach"

    return {
        "preventable": preventable,
        "collision_step": collision_step,
        "min_distance": min_distance,
        "max_pfs": max_pfs,
        "max_cfs": max_cfs,
        "criticality": criticality,
        "speed_at_collision": speed_at_collision,
        "collision_angle": collision_angle,
        "min_ttc": min_ttc,
        "max_braking_deficit": max_braking_deficit,
        "bd_at_trigger": bd_at_trigger,
        "trigger_step": trigger_step if trigger_step is not None else -1,
        "unpreventable_reason": unpreventable_reason,
        "timeseries": ts,
    }


# =============================================================================
# HDF5 extraction
# =============================================================================

def _decode_string(val):
    if isinstance(val, bytes):
        return val.decode("utf-8")
    if isinstance(val, np.ndarray):
        v = val.flat[0]
        if isinstance(v, bytes):
            return v.decode("utf-8")
        return str(v)
    return str(val)


def extract_agent_data(sim_group, agent_idx):
    centroid = sim_group["centroid"][agent_idx]
    yaw = sim_group["yaw"][agent_idx].flatten()
    ext = sim_group["extent"][agent_idx][0, :2].astype(np.float64)
    return centroid, yaw, ext


def find_agent_indices(sim_group, target_name_fallback=None):
    """Return (target_idx, adv_ts, agent_names) for one HDF5 episode group.

    Handles three formats:
      1. nuScenes: agent_name (N, T), target_name (1,), adv_idx (N, T)
      2. AD4CHE:   agent_name (N,),    target_name (1,), adv_idx (N, T)
      3. RounD:    no agent_name,      no target_name, adv_idx only
                   → infer target as nearest-to-adversary agent
    """
    # RounD fallback path
    if "agent_name" not in sim_group:
        N = sim_group["centroid"].shape[0]
        agent_names = [str(i) for i in range(N)]
        if "adv_idx" not in sim_group:
            return None, None, agent_names
        adv_idx_arr = sim_group["adv_idx"][:]
        adv_ts = adv_idx_arr[0] if adv_idx_arr.ndim >= 2 else adv_idx_arr
        adv_ts = adv_ts.astype(np.int32)
        adv_idx_last = int(adv_ts[-1])
        if not (0 <= adv_idx_last < N):
            return None, None, agent_names
        centroid = sim_group["centroid"][:]
        adv_xy = centroid[adv_idx_last]
        best_idx, best_d = None, float("inf")
        for i in range(N):
            if i == adv_idx_last:
                continue
            d = float(np.linalg.norm(centroid[i] - adv_xy, axis=-1).min())
            if d < best_d:
                best_d, best_idx = d, i
        return best_idx, adv_ts, agent_names

    agent_names_raw = sim_group["agent_name"]
    N = agent_names_raw.shape[0]
    if agent_names_raw.ndim == 1:
        agent_names = [_decode_string(agent_names_raw[i]) for i in range(N)]
    else:
        agent_names = [_decode_string(agent_names_raw[i, 0]) for i in range(N)]

    target_name = target_name_fallback
    if "target_name" in sim_group:
        target_name = _decode_string(sim_group["target_name"][0])

    if target_name is None:
        return None, None, agent_names

    target_idx = next((i for i, nm in enumerate(agent_names) if nm == target_name), None)
    if target_idx is None:
        target_idx = next(
            (i for i, nm in enumerate(agent_names) if target_name in nm or nm in target_name),
            None,
        )
    if target_idx is None:
        return None, None, agent_names

    if "adv_idx" not in sim_group:
        return target_idx, None, agent_names

    adv_idx_arr = sim_group["adv_idx"][:]
    adv_ts = adv_idx_arr[0] if adv_idx_arr.ndim >= 2 else adv_idx_arr
    adv_ts = adv_ts.astype(np.int32)

    adv_idx_last = int(adv_ts[-1])
    if not (0 <= adv_idx_last < N) or adv_idx_last == target_idx:
        return target_idx, None, agent_names

    return target_idx, adv_ts, agent_names


def extract_dynamic_adv_data(sim_group, adv_ts):
    """Build (T, 2) adv trajectory by selecting the right agent at each timestep."""
    centroid = sim_group["centroid"][:]
    yaw_raw  = sim_group["yaw"][:]
    extent_raw = sim_group["extent"][:]

    T = len(adv_ts)
    adv_pos = np.zeros((T, 2), dtype=np.float64)
    adv_yaw = np.zeros(T, dtype=np.float64)

    for t, idx in enumerate(adv_ts):
        adv_pos[t] = centroid[idx, t]
        adv_yaw[t] = float(np.asarray(yaw_raw[idx, t]).flat[0])

    last_idx = int(adv_ts[-1])
    adv_ext = extent_raw[last_idx, 0, :2].astype(np.float64)
    return adv_pos, adv_yaw, adv_ext


# =============================================================================
# nuScenes lane-width lookup for CC-JP-LaneAware
# =============================================================================

def build_scene_to_location(nusc) -> dict:
    """Return dict mapping scene name (e.g. 'scene-0297') → map location."""
    mapping = {}
    for scene in nusc.scene:
        log = nusc.get('log', scene['log_token'])
        mapping[scene['name']] = log['location']
    return mapping


def query_lane_width(nmap, x: float, y: float, fallback: float = 3.2) -> float:
    """Estimate lane width at world (x, y) via NuScenesMap closest-lane polygon.
    Returns `fallback` if no lane is found or estimate is implausible."""
    lane_token = nmap.get_closest_lane(x, y, radius=3.0)
    if not lane_token:
        return fallback
    try:
        lane_record = nmap.get('lane', lane_token)
        poly = nmap.extract_polygon(lane_record['polygon_token'])
        if poly is None or poly.is_empty:
            return fallback
        half_width = poly.boundary.distance(poly.centroid)
        lane_width = 2.0 * float(half_width)
        if lane_width < 2.0 or lane_width > 6.0:
            return fallback
        return lane_width
    except Exception:
        return fallback


def compute_dist_to_react(lane_width: float, adv_width: float,
                          ego_dist_to_lane: float = None) -> float:
    """R157 Annex 4 Appendix 3 dist_to_react formula:
        dist_to_react = ego_dist_to_lane + (lane_width - adv_width) / 2 - 0.72 - 0.375
    If ego_dist_to_lane is None, assume target is centered (ego = lane_width/2)."""
    if ego_dist_to_lane is None:
        ego_dist_to_lane = lane_width / 2.0
    return ego_dist_to_lane + (lane_width - adv_width) / 2.0 - 0.72 - 0.375


def load_nuscenes_context(nuscenes_root: str, nuscenes_version: str):
    """Load NuScenes + per-location maps. Returns (scene_to_loc, nmaps) or (None, None)."""
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.map_expansion.map_api import NuScenesMap
    except ImportError:
        print("[WARN] nuScenes devkit not installed; CC-JP-LaneAware will use fallback dist_to_react")
        return None, None

    if not os.path.isdir(nuscenes_root):
        print(f"[WARN] nuScenes root '{nuscenes_root}' not found; using fallback")
        return None, None

    print(f"Loading nuScenes scene→location index from {nuscenes_root} ...")
    t0 = time.time()
    try:
        nusc = NuScenes(version=nuscenes_version, dataroot=nuscenes_root, verbose=False)
    except Exception as e:
        print(f"[WARN] NuScenes load failed: {e}")
        return None, None
    scene_to_loc = build_scene_to_location(nusc)
    print(f"  Loaded {len(scene_to_loc)} scene mappings in {time.time()-t0:.1f}s")

    print("Loading nuScenes semantic maps ...")
    locations = {
        "boston-seaport",
        "singapore-onenorth",
        "singapore-queenstown",
        "singapore-hollandvillage",
    }
    nmaps = {loc: NuScenesMap(dataroot=nuscenes_root, map_name=loc) for loc in locations}
    print(f"  Loaded {len(nmaps)} maps")
    return scene_to_loc, nmaps


def precompute_dist_to_react(hdf5_path: str, scene_to_loc, nmaps,
                             fallback: float = 1.20):
    """Per-episode dist_to_react. Returns ep_key → (dtr, lane_width, adv_width, loc)."""
    result = {}
    with h5py.File(hdf5_path, "r") as f:
        for ep_key in f.keys():
            g = f[ep_key]
            scene_name = ep_key.split("_")[0]

            try:
                target_idx, adv_ts, _ = find_agent_indices(g)
                if target_idx is None or adv_ts is None:
                    result[ep_key] = (fallback, None, None, None)
                    continue
                centroid = g["centroid"]
                extent = g["extent"]
                target_pos0 = np.asarray(centroid[target_idx, 0], dtype=np.float64)
                if not np.all(np.isfinite(target_pos0)):
                    result[ep_key] = (fallback, None, None, None)
                    continue
                adv_idx_last = int(adv_ts[-1])
                adv_width = float(extent[adv_idx_last, 0, 1])
                if adv_width < 0.5 or adv_width > 4.0:
                    adv_width = 1.85
            except Exception:
                result[ep_key] = (fallback, None, None, None)
                continue

            loc = None
            lane_width = None
            if scene_to_loc is not None and nmaps is not None:
                loc = scene_to_loc.get(scene_name)
                if loc and loc in nmaps:
                    lane_width = query_lane_width(
                        nmaps[loc], float(target_pos0[0]), float(target_pos0[1]),
                    )

            if lane_width is None:
                result[ep_key] = (fallback, None, adv_width, None)
            else:
                dtr = compute_dist_to_react(lane_width, adv_width)
                result[ep_key] = (dtr, lane_width, adv_width, loc)
    return result


# =============================================================================
# Main
# =============================================================================

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nuscenes_root", default="/data/nuscenes")
    parser.add_argument("--nuscenes_version", default="v1.0-trainval")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--save_timeseries", action="store_true")
    parser.add_argument("--trust_hdf5_collision", action="store_true", default=True,
                        help="Trust HDF5 = collision-only (CARS default)")
    parser.add_argument("--no_trust_hdf5_collision", action="store_false",
                        dest="trust_hdf5_collision",
                        help="Force OBB re-check (for SafeSim-style save-all HDF5)")
    parser.add_argument("--dtr_fallback", type=float, default=1.20,
                        help="dist_to_react fallback when map lookup fails")


def run(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_timeseries:
        os.makedirs(os.path.join(args.output_dir, "timeseries"), exist_ok=True)

    scene_to_loc, nmaps = load_nuscenes_context(args.nuscenes_root, args.nuscenes_version)

    print(f"Pre-computing dist_to_react for episodes in {args.hdf5_path} ...")
    dtr_map = precompute_dist_to_react(
        args.hdf5_path, scene_to_loc, nmaps, fallback=args.dtr_fallback,
    )
    n_total = len(dtr_map)
    n_map_hit = sum(1 for v in dtr_map.values() if v[3] is not None)
    print(f"  {n_total} episodes, {n_map_hit} with map-based lane width "
          f"({100*n_map_hit/max(n_total,1):.1f}%)")

    ctrl_fsm  = FSMController(dt=args.dt)
    ctrl_ccjp = CCJPLaneAwareController(dt=args.dt)
    ctrl_rss  = RSSController(dt=args.dt)
    controllers = [ctrl_fsm, ctrl_ccjp, ctrl_rss]

    results_all = []
    with h5py.File(args.hdf5_path, "r") as f:
        ep_keys = sorted(list(f.keys()))
        print(f"Replaying {len(ep_keys)} episodes with 3 controllers ...")

        for ep_key in tqdm(ep_keys):
            g = f[ep_key]
            tgt_fallback = None
            m = re.search(r'_tgt([^_]+)_adv', ep_key)
            if m:
                tgt_fallback = m.group(1)
            try:
                target_idx, adv_ts, agent_names = find_agent_indices(
                    g, target_name_fallback=tgt_fallback,
                )
            except Exception:
                continue
            if target_idx is None or adv_ts is None:
                continue
            adv_idx_last = int(adv_ts[-1])

            try:
                target_pos, target_yaw, target_ext = extract_agent_data(g, target_idx)
                adv_pos, adv_yaw, adv_ext = extract_dynamic_adv_data(g, adv_ts)
            except Exception:
                continue

            if args.trust_hdf5_collision:
                raw_hit, raw_step, raw_min_d = True, None, float("nan")
            else:
                raw_hit, raw_step, raw_min_d = check_raw_collision(
                    target_pos, target_yaw, adv_pos, adv_yaw, target_ext, adv_ext,
                )

            dtr, lane_width, adv_width, loc = dtr_map.get(
                ep_key, (args.dtr_fallback, None, None, None),
            )
            ctrl_ccjp.set_dist_to_react(dtr)

            scene_id = ep_key.split("_")[0]
            target_name = agent_names[target_idx] if target_idx < len(agent_names) else ""
            adv_name = agent_names[adv_idx_last] if adv_idx_last < len(agent_names) else ""

            for ctrl in controllers:
                try:
                    result = replay_episode(
                        target_pos, target_yaw, adv_pos, adv_yaw,
                        target_ext, adv_ext, ctrl, dt=args.dt,
                    )
                except Exception as e:
                    print(f"  {ep_key}/{ctrl.name}: replay error {e}")
                    continue

                row = {
                    "episode": ep_key,
                    "scene_id": scene_id,
                    "controller": ctrl.name,
                    "target_name": target_name,
                    "adv_name": adv_name,
                    "preventable": result["preventable"],
                    "collision_step": result["collision_step"],
                    "min_distance": result["min_distance"],
                    "max_pfs": result["max_pfs"],
                    "max_cfs": result["max_cfs"],
                    "criticality": result["criticality"],
                    "speed_at_collision": result["speed_at_collision"],
                    "collision_angle_rad": result["collision_angle"],
                    "min_ttc": result["min_ttc"],
                    "max_braking_deficit": result["max_braking_deficit"],
                    "bd_at_trigger":       result["bd_at_trigger"],
                    "trigger_step":        result["trigger_step"],
                    "unpreventable_reason": result["unpreventable_reason"],
                    "raw_collision": raw_hit,
                    "raw_collision_step": raw_step,
                    "raw_min_distance": raw_min_d,
                    "lane_width_m": lane_width if lane_width is not None else -1,
                    "dist_to_react_m": dtr,
                    "map_location": loc or "fallback",
                }
                results_all.append(row)

                if args.save_timeseries:
                    ts_path = os.path.join(
                        args.output_dir, "timeseries",
                        f"{ep_key}_{ctrl.name}.npz",
                    )
                    np.savez_compressed(ts_path, **result["timeseries"])

    if not results_all:
        print("\nNo results.")
        return

    csv_path = os.path.join(args.output_dir, "replay_results_laneaware.csv")
    with open(csv_path, "w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(results_all[0].keys()))
        w.writeheader()
        w.writerows(results_all)
    print(f"\nResults saved: {csv_path}")

    import pandas as pd
    df = pd.DataFrame(results_all)
    n_eps = df[df.controller == "FSM"].shape[0]
    print(f"\n=== Summary ({n_eps} episodes) ===")
    for ctrl_name in ["FSM", "CC-JP-LaneAware", "RSS"]:
        sub = df[df.controller == ctrl_name]
        n = len(sub)
        if n == 0:
            continue
        unprev = (~sub["preventable"]).sum()
        print(f"  {ctrl_name:<17} unpreventable: {unprev:>4}/{n}  "
              f"({100*unprev/n:.1f}%)")
        if ctrl_name == "FSM":
            for crit in ["Easy", "Medium", "Hard"]:
                n_c = (sub["criticality"] == crit).sum()
                print(f"    {crit}: {n_c}")


if __name__ == "__main__":
    _parser = argparse.ArgumentParser(
        description="Lane-aware CCDM replay (FSM/CC-JP-LaneAware/RSS) — self-contained",
    )
    add_args(_parser)
    run(_parser.parse_args())
