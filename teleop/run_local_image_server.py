#!/usr/bin/env python3
"""Run teleimager image_server on the host with logging_mp compatibility."""

from __future__ import annotations

import logging_mp

if not hasattr(logging_mp, "basicConfig") and hasattr(logging_mp, "basic_config"):
    logging_mp.basicConfig = logging_mp.basic_config
if not hasattr(logging_mp, "getLogger") and hasattr(logging_mp, "get_logger"):
    logging_mp.getLogger = logging_mp.get_logger

from teleimager.image_server import main


if __name__ == "__main__":
    main()
