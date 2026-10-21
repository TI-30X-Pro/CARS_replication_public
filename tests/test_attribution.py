"""Tests for safety drivers, collision replay, and attribution metrics."""

import numpy as np
import pytest

from cars.attribution.drivers import CCJP, FSM, RSS
from cars.attribution.metrics import h_crit
from cars.attribution.replay import replay_episode


def state(gap=12.0, v=10.0, u=10.0, lat=0.0, a=0.0):
    hl = 2.25
    return {"distance": max(gap, 0.0), "d_lon_signed": gap, "target_speed": v, "adv_speed": u,
            "target_accel": a, "lateral_offset": lat, "half_width_target": 0.95, "half_width_adv": 0.95,
            "half_length_target": hl, "half_length_adv": hl, "adv_v_lat": 0.0}


def test_fsm_fuzzy_values():
    s = state(gap=12.0, v=10.0, u=10.0)
    FSM()(s)
    d_safe = 10 * 0.75 + 100 / 8 - 100 / 14 + 2
    d_unsafe = 10 * 0.75 + 100 / 12 - 100 / 14
    assert s["_pfs"] == pytest.approx((d_safe - 10.0) / (d_safe - d_unsafe))
    assert s["_cfs"] == 0.0                      # not closing: no critical risk


def test_fsm_reaction_window_then_jerk_ramp():
    fsm, out = FSM(), []
    for _ in range(14):
        out.append(fsm(state(gap=5.0, v=12.0, u=0.0)))
    first = next(i for i, a in enumerate(out) if a > 0)
    assert first == 9                            # triggered at 0, then 0.75 s counted down in 0.1 s steps
    assert out[first] == pytest.approx(1.265) and out[first + 1] == pytest.approx(2.53)
    assert max(out) <= 6.0


def test_no_braking_when_adversary_not_ahead():
    for drv in (FSM(), CCJP(), RSS()):
        assert drv(state(gap=-1.0, v=15.0, u=0.0)) == 0.0


def test_ccjp_needs_short_ttc():
    assert CCJP()(state(gap=30.0, v=10.0, u=0.0)) == 0.0          # TTC 3 s
    assert CCJP()(state(gap=15.0, v=10.0, u=0.0)) == pytest.approx(0.4)   # TTC 1.5 s: release phase


def test_rss_safe_distance():
    rss = RSS()
    expected = 10 * 0.75 + 3 * 0.75 ** 2 / 2 + (10 + 0.75 * 3) ** 2 / 12 - 5 ** 2 / 12
    assert rss.safe_lon(10.0, 5.0) == pytest.approx(expected)


class NoBrake:
    name = "none"
    triggered = False

    def reset(self):
        pass

    def __call__(self, s):
        return 0.0


def straight_scene(gap0):
    """Target at 10 m/s on a straight road, adversary standing still gap0 m ahead, 0.5 m aside."""
    t = np.arange(60) * 0.1
    target = np.stack([10.0 * t, np.zeros_like(t)], 1)
    adv = np.tile([gap0, 0.5], (60, 1))
    ext = np.array([4.5, 2.0])
    return target, np.zeros(60), adv, np.zeros(60), ext, ext


def test_replay_without_braking_keeps_the_collision():
    # S23: with no braking the replayed target is the logged one, so it collides at the
    # same step as the logged trajectory (front corners meet after 24.5 m, step 25)
    from cars.attribution.replay import _raw_collision
    scene = straight_scene(30.0)
    r = replay_episode(*scene, NoBrake())
    hit, step, _ = _raw_collision(scene[0], scene[1], scene[2], scene[3], scene[4], scene[5])
    assert hit and not r["preventable"]
    assert r["collision_step"] == step == 25


def test_fsm_avoids_a_stopped_car_with_room():
    r = replay_episode(*straight_scene(45.0), FSM())
    assert r["preventable"] and r["criticality"] in ("Medium", "Hard")


def test_h_crit_matches_the_paper():
    # Table S2, CARS: 362 FSM-avoidable collisions, Hard / Medium / Easy = 26 / 207 / 129
    tiers = ["Hard"] * 26 + ["Medium"] * 207 + ["Easy"] * 129
    assert h_crit(tiers) == pytest.approx(0.798, abs=5e-4)
