# logger.py
import os
import sys
from pathlib import Path
from loguru import logger


class LoguruCompatHandler:
    """兼容标准logging模块的格式化方式"""

    def __init__(self, logger_instance):
        self.logger = logger_instance

    def debug(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.logger.debug(msg, **kwargs)

    def info(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.logger.info(msg, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.logger.warning(msg, **kwargs)

    def error(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.logger.error(msg, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if args:
            msg = msg % args
        self.logger.critical(msg, **kwargs)


def setup_logger():
    """
    配置loguru日志记录器，包含异常捕获机制

    Returns:
        LoguruCompatHandler: 兼容处理程序
    """
    try:
        PROJECT_ROOT = find_project_root()
        os.chdir(PROJECT_ROOT)
        # 确保logs目录存在
        os.makedirs("logs", exist_ok=True)

        # 移除默认的控制台输出
        logger.remove()

        # # 添加控制台输出
        # logger.add(
        #     sys.stdout,
        #     format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        #     level="INFO"
        # )

        # 添加文件输出，按天轮转
        logger.add(
            "logs/api_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="DEBUG",
            rotation="00:00",  # 每天00:00轮转
            retention="30 days",  # 保留30天的日志
            compression="zip",  # 压缩旧日志文件
            encoding="utf-8",
            catch=True  # 捕获日志记录过程中的异常，避免因文件问题导致程序崩溃
        )

        # 返回兼容处理程序
        return LoguruCompatHandler(logger)

    except Exception as e:
        # 如果日志配置失败，静默处理，只使用控制台输出
        try:
            logger.remove()
            logger.add(
                sys.stdout,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level="INFO"
            )
            # 即使文件日志失败，也返回控制台日志的处理程序
            return LoguruCompatHandler(logger)
        except:
            # 如果连控制台日志都无法配置，静默忽略
            return None
def find_project_root(start_dir=os.getcwd()):  # 寻找项目根路径
    current_dir = start_dir
    if getattr(sys, 'frozen', False):
        # 打包环境：返回可执行文件所在目录
        return Path(sys.executable).parent
    while True:
        if "main.py" in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # Reached the root directory
            raise FileNotFoundError("Could not find main.py in any parent directory.")
        current_dir = parent_dir

# 创建全局logger实例
app_logger = setup_logger()