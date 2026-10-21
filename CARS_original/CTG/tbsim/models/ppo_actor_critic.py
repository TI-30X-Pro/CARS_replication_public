from tbsim.utils.geometry_utils import transform_points_tensor
import torch.nn as nn
import torch
from tianshou.data import to_torch
from torch.distributions import Independent
import os, json, torch, pytorch_lightning as pl
from datetime import datetime



@torch.no_grad()
def probe_from_obs_dict(actor, obs_dict, logger=None, step=0, max_points=8192, S=6):
    """Log GMM head diagnostics (sigma / mixture / sample diversity) to W&B."""
    # 1) Head parameters: log-sigma distribution and saturation rates.
    raw_logits, mu, log_sigma = actor.peek_head(obs_dict)
    ls = log_sigma.flatten()
    if ls.numel() > max_points:
        idx = torch.randint(ls.numel(), (max_points,), device=ls.device)
        ls = ls[idx]
    sig = torch.exp(ls).cpu().numpy()
    sigma_hit_low  = float((sig < 0.01).mean())
    sigma_hit_high = float((sig > 2.00).mean())
    ls_mean = float(ls.mean().item())
    ls_std  = float(ls.std().item())

    # 2) GMM mixture health: entropy, peak weight, effective component count.
    pi = torch.softmax(raw_logits, dim=-1)
    pi_entropy = float((-(pi.clamp_min(1e-12).log() * pi).sum(-1)).mean().item())
    pi_max     = float(pi.max(-1).values.mean().item())
    pi_neff    = float((1.0 / (pi * pi).sum(-1)).mean().item())

    # 3) Sampling diversity: sample the same obs S times and report std.
    dist, _ = actor(obs_dict)
    samples = dist.sample((S,))
    traj_div_mean = float(samples.float().std(dim=0).mean().item())

    to_log = {
        "diag/log_sigma_mean": ls_mean,
        "diag/log_sigma_std":  ls_std,
        "diag/sigma_hit_low":  sigma_hit_low,
        "diag/sigma_hit_high": sigma_hit_high,
        "diag/pi_entropy_mean": pi_entropy,
        "diag/pi_max_mean":     pi_max,
        "diag/pi_neff_mean":    pi_neff,
        "diag/traj_div_mean":   traj_div_mean,
    }
    if logger is not None and getattr(logger, "wandb_run", None) is not None:
        import wandb
        to_log["diag/log_sigma_hist"] = wandb.Histogram(ls.cpu().numpy())
        logger.wandb_run.log(to_log, step=step)



def build_save_best_fn(save_dir, eval_cfg, wandb_run=None, max_keep=3):
    os.makedirs(save_dir, exist_ok=True)

    # Chronologically-ordered (policy_path, critic_path, timestamp) tuples.
    saved_checkpoints = []

    def _atomic_torch_save(obj, path: str):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def _mk_ckpt_from_ts_policy(ts_policy):
        if not hasattr(ts_policy, "actor"):
            raise RuntimeError("ts_policy has no attribute `actor`")
        actor = ts_policy.actor
        policy_model = getattr(actor, "ppo_diffuser", None)
        if policy_model is None:
            raise RuntimeError("actor has no attribute `ppo_diffuser`")

        sd = policy_model.state_dict()
        sd_prefixed = {f"nets.policy.{k}": v.detach().cpu() for k, v in sd.items()}

        with torch.no_grad():
            param_l2 = 0.0
            for p in policy_model.parameters():
                param_l2 += float(p.detach().float().norm().item())

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt = {
            "state_dict": sd_prefixed,
            "pytorch-lightning_version": pl.__version__,
            "epoch": 0,
            "global_step": 0,
            "lr_schedulers": [],
            "optimizer_states": [],
            "datamodule_hyper_parameters": {},
            "hyper_parameters": {},
            "training_timestamp": ts,
            "config_summary": {
                "learning_rate": eval_cfg.ppo['learning_rate'],
                "gamma": eval_cfg.ppo['gamma'],
                "clip_ratio": eval_cfg.ppo['clip_ratio'],
                "seed": eval_cfg.seed,
            },
            "debug_param_l2": param_l2,
        }
        return ckpt, ts

    def _cleanup_old_checkpoints():
        """Drop checkpoint pairs older than `max_keep`."""
        while len(saved_checkpoints) > max_keep:
            policy_path, critic_path, ts = saved_checkpoints.pop(0)
            for path in [policy_path, critic_path]:
                if path and os.path.exists(path):
                    os.remove(path)
            print(f"[PPO] removed old checkpoint pair: {ts}")

    def save_best_fn(ts_policy):
        ckpt, ts = _mk_ckpt_from_ts_policy(ts_policy)

        best_path = os.path.join(save_dir, f"policy_best_{ts}.ckpt")
        critic_path = None

        _atomic_torch_save(ckpt, best_path)
        print(f"[PPO] saved BEST -> {best_path} (param_l2={ckpt['debug_param_l2']:.3f})")

        critic = getattr(ts_policy, "critic", None)
        if critic is not None:
            critic_path = os.path.join(save_dir, f"critic_best_{ts}.pth")
            torch.save(
                {"critic": critic.state_dict(), "saved_at": ckpt["training_timestamp"]},
                critic_path,
            )
            print(f"[PPO] saved CRITIC -> {critic_path}")

        saved_checkpoints.append((best_path, critic_path, ts))
        _cleanup_old_checkpoints()

        if wandb_run is not None:
            import wandb
            art = wandb.Artifact(f"ppo_rl_{ts}", type="model")
            art.add_file(best_path)
            if critic_path:
                art.add_file(critic_path)
            wandb_run.log_artifact(art)

    return save_best_fn


def build_save_checkpoint_fn(save_dir, policy, eval_cfg):
    """Build a periodic save callback that overwrites a single "latest" file.

    Signature: ``f(epoch, env_step, gradient_step) -> str``, invoked by the
    tianshou logger.
    """
    os.makedirs(save_dir, exist_ok=True)
    policy_path = os.path.join(save_dir, "policy_latest.ckpt")
    critic_path = os.path.join(save_dir, "critic_latest.pth")

    def _atomic_torch_save(obj, path: str):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def save_checkpoint_fn(epoch: int, env_step: int, gradient_step: int) -> str:
        actor = policy.actor
        policy_model = getattr(actor, "ppo_diffuser", None)
        if policy_model is None:
            raise RuntimeError("actor has no attribute `ppo_diffuser`")

        sd = policy_model.state_dict()
        sd_prefixed = {f"nets.policy.{k}": v.detach().cpu() for k, v in sd.items()}
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        ckpt = {
            "state_dict": sd_prefixed,
            "pytorch-lightning_version": pl.__version__,
            "epoch": epoch,
            "global_step": gradient_step,
            "env_step": env_step,
            "lr_schedulers": [],
            "optimizer_states": [],
            "datamodule_hyper_parameters": {},
            "hyper_parameters": {},
            "training_timestamp": ts,
            "config_summary": {
                "learning_rate": eval_cfg.ppo["learning_rate"],
                "gamma": eval_cfg.ppo["gamma"],
                "clip_ratio": eval_cfg.ppo["clip_ratio"],
                "seed": eval_cfg.seed,
            },
        }
        _atomic_torch_save(ckpt, policy_path)

        critic = getattr(policy, "critic", None)
        if critic is not None:
            torch.save(
                {"critic": critic.state_dict(), "saved_at": ts, "epoch": epoch},
                critic_path,
            )

        print(f"[PPO] saved LATEST -> {policy_path} (epoch={epoch}, env_step={env_step})")
        return policy_path

    return save_checkpoint_fn