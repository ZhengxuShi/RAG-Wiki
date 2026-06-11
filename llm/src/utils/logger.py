import os
import logging


def setup_logger(log_filename):
    """ 配置日志记录 """
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)  # 确保日志目录存在

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    return logging.getLogger()
