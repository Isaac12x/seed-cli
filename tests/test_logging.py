import logging
from seed_cli.logging import setup_logging, get_logger


def test_default_level():
    setup_logging()
    log = get_logger()
    assert log.level == logging.WARNING


def test_verbose_level():
    setup_logging(verbose=True)
    log = get_logger()
    assert log.level == logging.INFO


def test_debug_level():
    setup_logging(debug=True)
    log = get_logger()
    assert log.level == logging.DEBUG


def test_no_duplicate_handlers():
    setup_logging()
    log = get_logger()
    handlers_before = len(log.handlers)
    setup_logging()
    handlers_after = len(log.handlers)
    assert handlers_before == handlers_after


def test_child_logger_name():
    log = get_logger("planner")
    assert log.name.endswith("seed_cli.planner")
