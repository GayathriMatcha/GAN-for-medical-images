# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 18:28:52 2022

@author: hp
"""

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import tensorflow as tf


class Logger(object):
    """Tensorboard logger."""

    def __init__(self, log_dir):
        """Initialize summary writer."""
#         self.writer = tf.summary.FileWriter(log_dir)
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Add scalar summary."""
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])

        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
#         self.writer.add_summary(summary, step)
# import logging
# import time
# from datetime import timedelta


# class LogFormatter():

#     def __init__(self):
#         self.start_time = time.time()

#     def format(self, record):
#         elapsed_seconds = round(record.created - self.start_time)

#         prefix = "%s - %s - %s" % (
#             record.levelname,
#             time.strftime('%x %X'),
#             timedelta(seconds=elapsed_seconds)
#         )
#         message = record.getMessage()
#         message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
#         return "%s - %s" % (prefix, message)


# def create_logger(filepath):
#     """
#     Create a logger.
#     """
#     # create log formatter
#     log_formatter = LogFormatter()

#     # create file handler and set level to debug
#     if filepath is not None:
#         file_handler = logging.FileHandler(filepath,'a')
#         file_handler.setLevel(logging.DEBUG)
#         file_handler.setFormatter(log_formatter)

#     # create console handler and set level to info
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     console_handler.setFormatter(log_formatter)

#     # create logger and set level to debug
#     logger = logging.getLogger()
#     logger.handlers = []
#     logger.setLevel(logging.INFO)
#     logger.propagate = False
#     if filepath is not None:
#         logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     # reset logger elapsed time
#     def reset_time():
#         log_formatter.start_time = time.time()
#     logger.reset_time = reset_time

#     return logger