import numpy as np


class CollectDistributionParameter:
    """
    This callback can be used to collect the values of a parameter
    (e.g. learning rate) during a run of the agent.

    """
    def __init__(self, distribution):

        self._distribution = distribution
        self._p = list()

    def __call__(self, **kwargs):
        """
        Add the parameter value to the parameter values list.

        Args:
            **kwargs (dict): empty dictionary.

        """

        value = self._distribution.get_parameters()
        if isinstance(value, np.ndarray):
            value = np.array(value)
        self._p.append(value)

    def get_values(self):

        return self._p

    def reset(self):

        self._p = list()