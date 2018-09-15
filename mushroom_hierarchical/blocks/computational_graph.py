import numpy as np
from mushroom_hierarchical.utils.topological_sort import topological_sort
from mushroom_hierarchical.blocks.control_block import ControlBlock


class ComputationalGraph(object):
    """
    This class implements the computational graph for hierarchical learning.

    """
    def __init__(self, blocks, model):

        self.ordered = topological_sort(blocks)
        self.model = model
        self.state = list()
        self.reward = None
        self.absorbing = False
        self.last = False
        self.step_counter = 0

    def call_blocks(self, learn_flag, render):
        """
        executes the blocks in the diagram in the provided order. Always starts from the model.

        """
        action = self.ordered[-1].last_output
        #print action
        self.state, self.reward, self.absorbing, _ = self.model.step(action)
        self.step_counter += 1
        self.last = self.step_counter >= self.model.info.horizon or self.absorbing
        self.ordered[0].last_output = self.state
        self.ordered[1].last_output = np.array([self.reward])
        self.ordered[2].last_output = action
        #print 'ENV STATE, REW', self.state, self.reward
        for block in self.ordered:
            #print('NAME  :',block.name)
            inputs = list()
            alarms = list()
            for input_block in block.input_connections:
                #print('INPUTS: ', input_block.name)

                inputs.append(input_block.last_output)

            for i in inputs:
                if i is not None:
                    if np.any(np.isnan(i)):
                        print(inputs)
                        exit()
            if not block.alarm_connections:
                alarms.append(True)
            else:
                for alarm_connection in block.alarm_connections:
                    alarms.append(alarm_connection.alarm_output)
            if block.reward_connection is None:
                reward = None
                #print('REWARDS: None')
            else:
                reward = block.reward_connection.last_output[0]
                #print('REWARDS: ', block.reward_connection.name)
            block(inputs=inputs, reward=reward, absorbing=self.absorbing, last=self.last, learn_flag=learn_flag, alarms=alarms)

        if render:
            self.model.render()
        return self.absorbing, self.last

    def reset(self):
        self.state = self.model.reset()
        self.ordered[0].last_output = self.state
        self.ordered[1].last_output = None
        self.ordered[2].last_output = None
        #print 'ENV RESET STATE, REW', self.state, self.reward

        for block in self.ordered:
            #print('NAME :', block.name)
            inputs = list()
            for input_block in block.input_connections:
                inputs.append(input_block.last_output)
            #print('INPUTS   :', inputs)
            for i in inputs:
                if i is not None:
                    if np.any(np.isnan(i)):
                        exit()
            block.reset(inputs=inputs)
        self.step_counter = 0

    def get_sample(self):
        state = self.ordered[0].last_output
        action = self.ordered[-1].last_output
        rew_last = self.ordered[1].last_output
        abs = self.absorbing
        last = self.last
        return state, action, rew_last, abs, last

    def init(self):
        for block in self.ordered:
            block.init()
        self.step_counter = 0

    def stop(self):
        self.model.stop()
        for block in self.ordered:
            if isinstance(block, ControlBlock):
                block.stop()
