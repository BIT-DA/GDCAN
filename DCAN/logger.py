# -*- coding: utf-8 -*-

import os
import time
import logging
from logging import handlers


class Logger(object):

    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # Log level

    def __init__(self, logroot, filename, level='info', when='D', fmt='%(message)s'):

        if not os.path.exists(logroot):
            os.makedirs(logroot)

        filename = logroot + time.strftime('%Y-%m-%d %H:%M:%S') + ' ' + filename + '.log'
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)  # Set the log format
        self.logger.setLevel(self.level_relations.get(level))  # Set the log level
        sh = logging.StreamHandler()  # Output to the screen
        sh.setFormatter(format_str)

        # Write a processor to a file that generates the file automatically at specified intervals
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger(logroot='log/', filename='test', level='debug')
    log.logger.debug('Logger test.')
