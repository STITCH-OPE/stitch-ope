#!/usr/bin/env python3
"""
Sanity check for MuJoCo + mujoco-py + d4rl (NO datasets needed).

- Verifies imports (mujoco_py, gym, d4rl)
- Creates/steps a MuJoCo gym env ("Hopper-v2")
- Creates/steps a d4rl-registered env ("hopper-medium-v2") WITHOUT calling get_dataset()

Usage examples:
  MUJOCO_GL=egl   python sanity_d4rl_mujoco.py   # GPU/headless
  MUJOCO_GL=osmesa python sanity_d4rl_mujoco.py  # CPU/headless
  python sanity_d4rl_mujoco.py --backend egl --steps 5
"""

import argparse
import os
import sys
import traceback

def pick_backend(backend: str) -> str:
    if backend and backend.lower() != "auto":
        os.environ["MUJOCO_GL"] = backend.lower()
        return backend.lower()
    # auto: prefer existing, else egl -> osmesa -> glfw
    if "MUJOCO_GL" in os.environ and os.environ["MUJOCO_GL"]:
        return os.environ["MUJOCO_GL"]
    for b in ("egl", "osmesa", "glfw"):
        try:
            os.environ["MUJOCO_GL"] = b
            return b
        except Exception:
            continue
    return os.environ.get("MUJOCO_GL", "")

def step_env(env_id: str, steps: int) -> None:
    import gym
    print(f"\n[env] making {env_id!r}")
    env = gym.make(env_id)
    print(f"[env] {env_id} made; action_space={env.action_space}, observation_space={env.observation_space}")
    obs = env.reset()
    total = 0.0
    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total += float(reward)
        if done:
            obs = env.reset()
    print(f"[env] {env_id} stepped {steps} times; cumulative_reward≈{total:.3f}")
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", help="MuJoCo GL backend: auto|egl|osmesa|glfw")
    parser.add_argument("--steps", type=int, default=10, help="Steps per env")
    parser.add_argument("--skip-d4rl", action="store_true", help="Skip the d4rl env check")
    args = parser.parse_args()

    backend = pick_backend(args.backend)
    print(f"[gl] MUJOCO_GL={backend or '(unset)'}")

    # --- imports ---
    try:
        import mujoco_py  # noqa: F401
        print("[import] mujoco_py: OK")
    except Exception as e:
        print("[import] mujoco_py: FAIL")
        traceback.print_exc()
        sys.exit(1)

    try:
        import gym  # noqa: F401
        print("[import] gym: OK")
    except Exception as e:
        print("[import] gym: FAIL")
        traceback.print_exc()
        sys.exit(1)

    # d4rl registers env IDs like 'hopper-medium-v2' upon import
    try:
        import d4rl  # noqa: F401
        print("[import] d4rl: OK")
        try:
            import d4rl.gym_mujoco  # older/original D4RL
        except Exception:
            try:
                import d4rl.mujoco  # Farama fork sometimes uses this path
            except Exception:
                pass
    except Exception as e:
        if args.skip_d4rl:
            print("[import] d4rl: SKIPPED by flag")
            d4rl = None
        else:
            print("[import] d4rl: FAIL")
            traceback.print_exc()
            sys.exit(1)

    # --- MuJoCo gym env (no dataset) ---
    try:
        step_env("Hopper-v2", steps=args.steps)
    except Exception:
        print("[run] Hopper-v2: FAIL")
        traceback.print_exc()
        sys.exit(2)

    # --- d4rl env (registered), but we do NOT call get_dataset() ---
    if not args.skip_d4rl:
        try:
            step_env("hopper-medium-v2", steps=args.steps)
        except Exception:
            print("[run] hopper-medium-v2: FAIL (note: should run without datasets; check GL backend & mujoco)")
            traceback.print_exc()
            sys.exit(3)

    print("\n[done] sanity OK.")
    sys.exit(0)

if __name__ == "__main__":
    main()
