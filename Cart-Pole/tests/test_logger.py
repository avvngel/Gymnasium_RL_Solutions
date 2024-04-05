#!/usr/bin/env python3

from log_controller import *

logger = Logger()
logger.set_verbosity(2)
msg = "Hello World"
logger(msg)
