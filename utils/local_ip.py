import socket


def get_local_ip():
    """取得區網 IP

    Returns:
        ip (_RetAddress): 區網 IP
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    print("IP：{}".format(ip))
    s.close()

    return ip
