def get_vocab(text):
    return sorted(list(set(text)))

def flatten_dict(d: dict, prefix: str = "") -> dict:
    out = {}

    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, name))
        else:
            out[name] = value

    return out