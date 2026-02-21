import logging, sys
from pathlib import Path
from typing import Union


#ログを出力する設定を作る
def build_logger(
    log_dir: Union[str, Path],
    log_file: str,
    *,
    console_output: bool = False,
    level: int = logging.INFO,
    overwrite: bool = False,
) -> logging.Logger:
    """
    Configure the logger.

    Args:
        log_dir (Union[str, Path]): Directory to save the log file.
        log_file (str): Name of the log file.
        console_output (bool, optional): Whether to output logs to console.
        level (int, optional): Logging level. Default is logging.INFO.
        overwrite (bool, optional): Whether to overwrite existing log files. Default is False.

    Returns:
        logging.Logger: Configured logger
    """

    log_dir = Path(log_dir).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if "." not in Path(log_file).name:
        log_file += ".log"

    full_path = log_dir / log_file

    logger_name = full_path.as_posix()
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    mode = "w" if overwrite else "a"

    # file handler
    file_handler = logging.FileHandler(full_path, mode=mode, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger