#!/usr/bin/env python3

class Logger():

    def __init__(self):
        self.verbosity = 0
        self.logger = self.null_logger

    def active_logger(self, msg, end = '\n'):
        print(msg, end = end)

    def normal_logger(self, msg, end = '\n'):
        print(msg, end = end)

    def null_logger(self, msg, end = '\n'):
        pass

    def set_verbosity(self, verbosity):
        self.verbosity = verbosity
        if verbosity == 0:
            self.logger = self.null_logger
        elif verbosity == 1:
            self.logger = self.normal_logger
        elif verbosity == 2:
            self.logger = self.active_logger

    def __call__(self, msg, msg_verbosity = 2, end = '\n'):
        if msg_verbosity <= self.verbosity:
            self.logger(msg, end = end)

