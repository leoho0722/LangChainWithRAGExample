from utils.config import ALLOWED_EXTENSIONS


def allowed_file(filename):
    """判斷上傳的檔案副檔名是否包含在允許的副檔名清單內

    Args:
        filename (str): 檔案名稱

    Returns:
        bool: 是否包含在允許的副檔名清單內
    """

    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
