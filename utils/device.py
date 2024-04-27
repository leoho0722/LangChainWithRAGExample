from torch import cuda, backends


def available_device() -> str:
    """取得可用的裝置

    Returns:
        str: 裝置名稱
    """

    device: str
    if cuda.is_available():
        device = "cuda"
    else:
        if backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return device
