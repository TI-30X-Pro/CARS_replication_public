"""Batching checks: neighbour padding when agents have no neighbours or short histories."""

import numpy as np
import torch

from cars.data.batch import EncodingConfig, collate, element, parse

CFG = EncodingConfig()


class TwoCars:
    """Two vehicles driving in line along x, `gap` metres apart, logged from `first`."""
    name, dt, location = "scene-test", 0.1, "boston-seaport"

    def __init__(self, gap, first=(0, 0)):
        self.agents = ["ego", "other"]
        self.types = np.array([int(CFG.agent_types[0])] * 2)
        self.first = np.array(first)
        self.last = np.array([200, 200])
        self.extent = np.array([[4.5, 1.9, 1.6]] * 2)
        self.states = []
        for x0, f in zip((0.0, gap), first):
            t = np.arange(f, 201) * self.dt
            s = np.zeros((len(t), 8))
            s[:, 0], s[:, 2], s[:, 3] = x0 + 5.0 * t, np.nan, 5.0
            self.states.append(s)

    def index(self, name):
        return self.agents.index(name)

    def state(self, i, ts):
        return self.states[i][ts - self.first[i]]

    def present(self, ts):
        return [i for i in range(2) if self.first[i] <= ts <= self.last[i]]


def observe(tracks, ts=141):
    return parse(collate([element(tracks, i, ts, None, CFG, get_map=False) for i in range(2)], CFG), CFG)


def test_neighbours_within_radius_are_kept():
    obs = observe(TwoCars(46.6))
    assert obs["all_other_agents_history_positions"].shape == (2, 1, CFG.history_steps + 1, 2)
    assert obs["all_other_agents_history_availability"].all()
    assert obs["neigh_indices"].tolist() == [[1.0], [0.0]]


def test_no_neighbours_gives_one_masked_slot():
    # scene-0847, shard 13: the last observation had the two agents 50.6 m apart
    obs = observe(TwoCars(50.6))
    assert obs["all_other_agents_history_positions"].shape == (2, 1, CFG.history_steps + 1, 2)
    assert obs["all_other_agents_curr_speed"].shape == (2, 1)
    assert not obs["all_other_agents_history_availability"].any()
    assert not obs["all_other_agents_future_availability"].any()
    assert obs["neigh_indices"].tolist() == [[-1.0], [-1.0]]
    assert obs["num_neigh"].tolist() == [0, 0]


def test_short_neighbour_histories_are_front_padded():
    # both agents logged from ts 130, so at ts 141 every history is 12 steps, not 21
    obs = observe(TwoCars(10.0, first=(130, 130)))
    avail = obs["all_other_agents_history_availability"]
    assert avail.shape == (2, 1, CFG.history_steps + 1)
    assert not avail[..., :-12].any() and avail[..., -12:].all()


def test_masked_slot_does_not_reach_the_agent_token():
    from cars.generation.nets import AgentsMHA, NeighborHistoryEncoder
    torch.manual_seed(0)
    obs = observe(TwoCars(50.6))
    norm = (np.array([1.0, -2.0, 5.0, 4.5, 1.9]), np.array([10.0, 8.0, 4.0, 1.0, 0.3]))
    tokens, mask = NeighborHistoryEncoder(64, norm)(
        obs["all_other_agents_history_positions"], obs["all_other_agents_history_speeds"],
        obs["all_other_agents_history_yaws"].squeeze(-1), obs["all_other_agents_history_extents"][:, :, 0, :2],
        obs["all_other_agents_history_availability"])
    mha = AgentsMHA(64, 4)
    center = torch.randn(2, 64)
    assert mask.all()
    assert torch.allclose(mha(center, tokens, mask), mha.norm(center))
