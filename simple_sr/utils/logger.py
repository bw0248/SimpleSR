import logging
import sys

LIB_LOGGER = "simple_sr"
RESULTS_LOGGER = "results"


def setup_logger(folder_name):
    """
    Basic logger to log debug messages.

    :param folder_name:
        Path to log folder where the log file will be saved.
    """
    logger = logging.getLogger(LIB_LOGGER)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "{name} - {levelname} - {message}", style="{"
    )

    # create handler that prints debug messages to file
    fh = logging.FileHandler(f"{folder_name}/log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # create handler for stdout
    std_out = logging.StreamHandler()
    std_out.setLevel(logging.INFO)
    std_out.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(std_out)

    # prevent spam from some modules
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    # setup training results logger
    res_fh = logging.FileHandler(f"{folder_name}/results_logfile")
    res_fh.setLevel(logging.INFO)
    res_stdout = logging.StreamHandler()
    res_stdout.setLevel(logging.INFO)

    res_logger = logging.getLogger(RESULTS_LOGGER)
    res_logger.setLevel(logging.INFO)
    res_logger.addHandler(res_fh)
    res_logger.addHandler(res_stdout)

    sys.excepthook = handle_exception


# handler for uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.getLogger(LIB_LOGGER).error("Uncaught exception",
                                        exc_info=(exc_type, exc_value, exc_traceback))


if __name__ == "__main__":
    pass

