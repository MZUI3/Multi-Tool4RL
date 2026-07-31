def train(env_id: str, **kwargs):
    msg = "A3C is not included in SB3. Clone or implement A3C under third_party and adapt this wrapper."
    print(msg)
    return {"status": "error", "msg": msg}
