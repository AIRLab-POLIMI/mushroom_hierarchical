from .block import Block
from .control_block import ControlBlock
import numpy as np

class MuxBlock(Block):
    """
    This class implements the multiplexer object of a computational graph for hierarchical learning.

    """
    def __init__(self, name=None):

        self.block_lists = list()
        self.first = list()

        super(MuxBlock, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):
        selector = inputs[0]
        state = inputs[1:]
        #print('MUX BLOCK-------')
        #print('selector in  : ', selector)
        #print('state in : ', state)
        alarms = list()

        #print('STEP ',selector)

        for i in range(len(self.block_lists)):
            if i == selector:
                selected_block_list = self.block_lists[i]
                if self.first[i]:
                    for block in selected_block_list:
                        block.reset(inputs=inputs[1:])
                    self.first[i] = False
                else:
                    for block in selected_block_list:
                        if block.reward_connection is not None:
                            reward = block.reward_connection.last_output[0]
                        else:
                            reward = None
                        if not block.alarm_connections:
                            alarms.append(True)
                        else:
                            for alarm_connection in block.alarm_connections:
                                alarms.append(alarm_connection.alarm_output)
                        block(inputs=state, reward=reward, absorbing=absorbing,
                              last=last, learn_flag=learn_flag, alarms=alarms)
                        state = block.last_output
            else:
                other_block_list = self.block_lists[i]
                for block in other_block_list:
                    block.alarm_output = False

        self.last_output = selected_block_list[-1].last_output

    def add_block_list(self, block_list):
        self.block_lists.append(block_list)
        self.first.append(True)

    def reset(self, inputs):
        selector = inputs[0]
        state = inputs[1:]
        #print('RESET ',selector)
        for i in range(len(self.block_lists)):
            if i == selector:
                selected_block_list = self.block_lists[i]
                for block in selected_block_list:
                    block.reset(inputs=state)
                self.first[i] = False
            else:
                self.first[i] = True

        self.last_output = selected_block_list[-1].last_output

    def init(self):
        self.last_output = None
        for block_list in self.block_lists:
            for block in block_list:
                block.init()
        for i in range(len(self.first)):
            self.first[i] = True

    def stop(self):
        for block_list in self.block_lists:
            for block in block_list:
                if isinstance(block, ControlBlock):
                    block.stop()