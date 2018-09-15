from mushroom.features.features import *
from mushroom.features.tiles import Tiles
from .block import Block


class DiscretizationBlock(Block):

    def __init__(self, low, high, n_tiles, name='discretizaation_block'):
        self.low = low*np.ones(shape=np.shape(n_tiles))
        self.high = high*np.ones(shape=np.shape(n_tiles))
        self.n_tiles = n_tiles
        self.tilings = Tiles.generate(n_tilings=1, n_tiles=self.n_tiles,
                                      low=self.low, high=self.high)
        super(DiscretizationBlock, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):
        state = np.concatenate(inputs, axis=0)

        index = list()
        for tile in self.tilings:
            index.append(tile(state))

        self.last_output = index

    def init(self):
        pass

    def reset(self, inputs):
        tilings = Tiles.generate(n_tilings=1, n_tiles=self.n_tiles,
                                 low=self.low, high=self.high)
        state = np.concatenate(inputs, axis=0)

        index = list()
        for tile in tilings:
            index.append(tile(state))
        self.last_output = index