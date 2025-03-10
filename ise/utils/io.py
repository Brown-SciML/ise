def check_type(obj, types):
    if not isinstance(obj, types):
        raise TypeError(f"Expected {types}, got {type(obj).__name__}")
    return 1