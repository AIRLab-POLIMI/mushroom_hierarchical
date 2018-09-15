from .block import Block


class PlaceHolder(Block):
    def __init__(self, name=None):
        super(PlaceHolder, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):
        pass

    def reset(self,inputs):
        pass

    def init(self):
        pass




