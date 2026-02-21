from ..base import Experiment
from .. import register

@register("training")
class TrainingExperiment(Experiment):

    def run(self):
        assert self.engine is not None, "Engine not built"
        self.engine.step(0, is_train=False) # Evaluate random initialization
        for epoch in range(1, self.cfg.num_epochs):
            self.engine.step(epoch, is_train=True)
            self.engine.step(epoch, is_train=False)