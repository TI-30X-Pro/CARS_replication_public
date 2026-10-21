"""Simulator geometry and episode-writing checks."""

import h5py
import numpy as np

from cars.sim.geometry import obb_collision, veh_veh_collision


def test_rollout_box_test():
    p = lambda x, y, h: np.array([[x, y, h]], dtype=np.float32)   # noqa: E731
    s = np.array([[4.0, 2.0]], dtype=np.float32)
    # a corner of the enlarged box 1 (5 x 2.3 m) inside box 2
    assert veh_veh_collision(p(0, 0, 0), p(3.0, 1.0, 0), s, s)[0] < 0
    assert veh_veh_collision(p(0, 0, 0), p(7.0, 1.0, 0), s, s)[0] > 0
    # The rollout test's blind spot: equal widths, driving in line, 2 m of overlap, but the
    # widened corners of box 1 all stick out sideways, so no collision is reported
    assert veh_veh_collision(p(0, 0, 0), p(3.0, 0, 0), s, s)[0] > 0


def test_replay_box_test_enlarges_both():
    # box 2 enlarged too: this corner counts in the replay test but not in the rollout test
    hit, d = obb_collision((0, 0), 0.0, (4.0, 2.0), (4.5, 1.0), 0.0, (4.0, 2.0))
    assert hit and d < 0
    p = lambda x, y: np.array([[x, y, 0.0]])   # noqa: E731
    assert veh_veh_collision(p(0, 0), p(4.5, 1.0), np.array([[4.0, 2.0]]), np.array([[4.0, 2.0]]))[0] >= 0


def test_episode_written_in_expected_layout(tmp_path):
    from cars.sim.rollout import EpisodeLog, write_episode
    n = 3
    log = EpisodeLog(scene_index=7)
    for step in range(2):
        obs = dict(centroid=np.zeros((n, 2), np.float32), yaw=np.zeros(n, np.float32),
                   extent=np.ones((n, 3), np.float32), world_from_agent=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
                   raster_from_world=np.tile(np.eye(3, dtype=np.float32), (n, 1, 1)),
                   agent_name=["ego", "a", "b"], scene_ids=["scene-0001"] * n)
        plan = np.zeros((n, 20 - step, 2), np.float32)
        log.log(obs, plan, plan[..., :1], {"adv_idx": np.full(n, 1, np.int32)})
    buf = log.serialize()
    buf["target_name"] = np.array(["b"], dtype=object)
    key = write_episode(tmp_path / "data.hdf5", "scene-0001", 21, 2, buf)
    assert key == "scene-0001_0021_t_2"
    with h5py.File(tmp_path / "data.hdf5", "r") as f:
        g = f[key]
        assert g["centroid"].shape == (n, 2, 2) and g["yaw"].shape == (n, 2)
        assert g["action_traj_positions"].shape == (n, 2, 20, 2)
        assert np.isnan(g["action_traj_positions"][0, 1, -1]).all()       # shorter plan padded with NaN
        assert g["agent_name"][2, 0].decode() == "b" and g["target_name"][0].decode() == "b"
        assert (g["scene_index"][:] == 7).all() and (g["track_id"][:, 0] == np.arange(n)).all()
