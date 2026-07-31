def train(env_id: str, script: str = None, **kwargs):
    """QMIX wrapper: tries to run QMIX training from a local pymarl/paper implementation.

    Looks for `third_party/pymarl` or `third_party/qmix` and runs a train script if found.
    """
    import os
    import subprocess
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party")
    candidates_repos = [os.path.join(base, d) for d in ("pymarl", "qmix", "pymarl-official")]
    repo = None
    for r in candidates_repos:
        if os.path.exists(r):
            repo = r
            break

    if not repo:
        msg = "No pymarl/qmix repo found under third_party. Run `bash setup.sh --all` to clone candidates."
        print(msg)
        return {"status": "error", "msg": msg}

    candidates = []
    if script:
        candidates.append(os.path.join(repo, script))
    # common train script locations
    candidates += [os.path.join(repo, p) for p in ("train.py", "scripts/train.py", "run_train.py")]

    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break

    if not found:
        msg = f"No train script found in {repo}. Inspect the repo for training entrypoints."
        print(msg)
        return {"status": "error", "msg": msg}

    cmd = ["python", found, "--env", env_id]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]

    print(f"Running QMIX train script: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=repo)
        return {"status": "ok", "msg": "QMIX training finished (exit 0)"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "msg": str(e)}
