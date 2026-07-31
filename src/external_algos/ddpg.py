def train(env_id: str, **kwargs):
    msg = "DDPG wrapper. SB3 does not include DDPG upstream; consider TD3 or use external implementations."
    print(msg)
    return {"status": "error", "msg": msg}
