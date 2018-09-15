from tqdm import tqdm
from mushroom_hierarchical.utils.dataset_manager import DatasetManager


class HierarchicalCore(object):
    """
    Implements the functions to run a generic computational graph.

    """
    def __init__(self, computational_graph, callbacks=None):

        self.computational_graph = computational_graph
        self.callbacks = callbacks if callbacks is not None else list()
        self._n_steps = None
        self._n_episodes = None

    def learn(self, n_steps=None, n_episodes=None, render=False, quiet=False, skip=False):
        return self._run(True, n_steps, n_episodes, render, quiet, skip)

    def evaluate(self, n_steps=None, n_episodes=None, render=False, quiet=False):
        return self._run(False, n_steps, n_episodes, render, quiet, False)

    def _run(self, learn_flag, n_steps, n_episodes, render, quiet, skip):
        dataset_manager = DatasetManager()
        self.computational_graph.init()
        assert (n_episodes is not None and n_steps is None) or (n_episodes is None and n_steps is not None)
        if n_steps is not None:
            last = True
            for step in tqdm(range(n_steps), dynamic_ncols=True,
                                   disable=quiet, leave=False):
                if last:
                    self.computational_graph.reset()
                    dataset_manager.add_first_sample(self.computational_graph.get_sample(), skip)
                absorbing, last = self.computational_graph.call_blocks(learn_flag=learn_flag, render=render)
                dataset_manager.add_sample(self.computational_graph.get_sample(), skip)

        else:
            for episode in tqdm(range(n_episodes), dynamic_ncols=True,
                                   disable=quiet, leave=False):
                self.computational_graph.reset()
                dataset_manager.add_first_sample(self.computational_graph.get_sample(), skip)
                last = False
                while not last:
                    absorbing, last = self.computational_graph.call_blocks(learn_flag=learn_flag, render=render)
                    dataset_manager.add_sample(self.computational_graph.get_sample(), skip)
        self.computational_graph.stop()
        return dataset_manager.dataset
