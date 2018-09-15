from .block import Block
import numpy as np
from copy import deepcopy


class ControlDataset():
    def __init__(self):
        self.dataset = list()
        self.s = None
        self.a = None

    def add_first(self, state, action):
        self.s = state
        self.a = action

    def add_next(self, next_state, reward, absorbing, last):
        sample = self.s, self.a, reward, next_state, absorbing, last
        self.dataset.append(sample)
        self.s = next_state

    def add_action(self, action):
        self.a = action

    def empty(self):
        del self.dataset[:]

    def get(self):
        return deepcopy(self.dataset)

    def __len__(self):
        return len(self.dataset)


class ControlBlock(Block):

    def __init__(self, name, agent, termination_condition=None, n_eps_per_fit=None,
                 n_steps_per_fit=None, callbacks=list()):
        self.agent = agent
        self.n_eps_per_fit = n_eps_per_fit
        self.n_steps_per_fit = n_steps_per_fit
        self.dataset = ControlDataset()
        self.horizon = self.agent.mdp_info.horizon
        self.gamma = self.agent.mdp_info.gamma
        self.callbacks = callbacks
        self.curr_episode_counter = 0
        self.ep_step_counter = 0
        self.need_reset = False
        self.mask = False

        if termination_condition is None:
            self.termination_condition = lambda x : False
        else:
            self.termination_condition = termination_condition

        super(ControlBlock, self).__init__(name=name)

    def _call(self, inputs, reward, absorbing, last, learn_flag):

        if self.mask:
            learn_flag = False

        state = np.concatenate(inputs, axis=0)
        self.ep_step_counter += 1

        if self.need_reset:
            if not last:
                self.reset(inputs)
        else:
            local_absorbing = self.termination_condition(state)
            local_last = local_absorbing or self.ep_step_counter >= self.horizon

            self.dataset.add_next(next_state=state, reward=reward,
                                absorbing=absorbing or local_absorbing,
                                last=last or local_last)
            self.need_reset = local_last

            if local_last or last:
                self.curr_episode_counter += 1

            if learn_flag and \
                (len(self.dataset) == self.n_steps_per_fit
                 or self.curr_episode_counter == self.n_eps_per_fit):

                dataset = self.dataset.get()

                self.agent.fit(dataset)
                self.dataset.empty()
                self.curr_episode_counter = 0

                for c in self.callbacks:
                    callback_pars = dict(dataset=dataset)
                    c(**callback_pars)
            if not last:
                next_action = self.agent.draw_action(state)
                self.dataset.add_action(next_action)
                self.last_output = next_action

            self.alarm_output = local_last

    def reset(self, inputs):

        state = np.concatenate(inputs, axis=0)
        self.agent.episode_start()
        action = self.agent.draw_action(state)
        self.dataset.add_first(state, action)
        self.ep_step_counter = 0
        self.alarm_output = False
        self.need_reset = False
        self.last_output = action

    def init(self):
        self.dataset.empty()
        self.ep_step_counter = 0
        self.curr_episode_counter = 0
        self.last_output = None
        self.need_reset = False
        self.alarm_output = False

    def stop(self):
        self.agent.stop()

    def set_mask(self):
        self.mask = True

    def unset_mask(self):
        self.mask = False
