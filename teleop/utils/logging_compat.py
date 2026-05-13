"""Compatibility helpers for logging_mp API variants."""

import logging_mp

if not hasattr(logging_mp, "basicConfig") and hasattr(logging_mp, "basic_config"):
    logging_mp.basicConfig = logging_mp.basic_config
if not hasattr(logging_mp, "getLogger") and hasattr(logging_mp, "get_logger"):
    logging_mp.getLogger = logging_mp.get_logger


def basic_config(*args, **kwargs):
    if hasattr(logging_mp, "basic_config"):
        return logging_mp.basic_config(*args, **kwargs)
    return logging_mp.basicConfig(*args, **kwargs)


def get_logger(name, *args, **kwargs):
    if hasattr(logging_mp, "get_logger"):
        return logging_mp.get_logger(name, *args, **kwargs)
    return logging_mp.getLogger(name, *args, **kwargs)
