def train(env_id: str, script: str = None, **kwargs):
    """Agent57 wrapper: look for third_party agent57 implementations and run their train script.

    Agent57 implementations vary; this wrapper attempts to run a provided script or common filenames.
    Pass `script` to point to the script path inside the repo if needed.
    """
    import os
    import subprocess
    repo_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "third_party", "agent57")
    if not os.path.exists(repo_root):
        msg = "Agent57 implementation not found. Run `bash setup.sh --all` to clone third_party repos."
        print(msg)
        return {"status": "error", "msg": msg}

    candidates = []
    if script:
        candidates.append(os.path.join(repo_root, script))
    candidates += [os.path.join(repo_root, p) for p in ("train.py", "agent57.py", "run.py", "main.py")]

    found = None
    for c in candidates:
        if os.path.exists(c):
            found = c
            break

    if not found:
        msg = f"No train script found for Agent57 in {repo_root}. Provide --script or inspect repo."
        print(msg)
        return {"status": "error", "msg": msg}

    cmd = ["python", found, "--env", env_id]
    for k, v in kwargs.items():
        cmd += [f"--{k}", str(v)]

    print(f"Running Agent57 script: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=repo_root)
        return {"status": "ok", "msg": "Agent57 training finished (exit 0)"}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "msg": str(e)}
