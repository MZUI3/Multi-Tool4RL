def train(env_id: str, script: str = None, **kwargs):
    """MuZero wrapper.

    Tries to find a local MuZero implementation under `third_party/muzero-general`.
    If found, attempts to run a plausible training script; otherwise prints instructions.
    Parameters:
      env_id: environment id (string) passed to the implementation when supported
      script: optional path to the training script inside the muzero repo
    """
    import os
    import subprocess
    repo_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party", "muzero-general")
    if not os.path.exists(repo_root):
        msg = "MuZero implementation not found. Run `bash setup.sh --all` to clone third_party repos."
        print(msg)
        return {"status": "error", "msg": msg}

    # Common candidate script names
    candidates = []
    if script:
        candidates.append(os.path.join(repo_root, script))
    candidates += [os.path.join(repo_root, p) for p in ("train.py", "muzero.py", "run_muzero.py", "main.py")]

    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break

    if not found:
        msg = f"No obvious train script found in {repo_root}. Inspect the repo and provide --script path."
        print(msg)
        return {"status": "error", "msg": msg}

    cmd = ["python", found, "--env", env_id]
    # pass through extra kwargs as --key value
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]

    print(f"Running MuZero train script: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=repo_root)
        return {"status": "ok", "msg": "MuZero training finished (exit 0)"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "msg": str(e)}
