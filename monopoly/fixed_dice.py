import logging
import numpy as np

from monopoly import config

logger = logging.getLogger(__name__)

class Dice:

    def __init__(self):

        self.roll_sum = None
        self.double = False
        self.double_counter = 0

    def roll(self, value1=0, value2=0):
        """Roll two fair six-sided die and store (1) the sum of the roll, (2) an indicator of whether it was a double
        roll and (3) a counter of the number of consecutive double rolls."""

        self.roll_sum = value1 + value2
        self.double = value1  == value2
        self.double_counter += self.double

        if config.verbose['dice']:
            logger.info('Roll a {die_1} and a {die_2}'.format(die_1=value1, die_2=value2))
