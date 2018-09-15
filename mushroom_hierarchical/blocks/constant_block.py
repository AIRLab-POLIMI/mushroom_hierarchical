from .block import Block


class ConstantBlock(Block):
    def __init__(self, constant, name=None):

        self.last_output = constant
        self.alarm_output = constant
        super(ConstantBlock, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):
        pass

    def reset(self, inputs):
        pass

    def init(self):
        pass
