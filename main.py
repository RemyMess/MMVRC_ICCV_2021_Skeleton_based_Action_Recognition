from src.preprocessing import *
from src.data_grabbing.data_grabber import DataGrabber
from src.algos import *


class SkeletonBasedActionRecognistion:
    def __init__(self):
        self.data = DataGrabber().getAllData()

    def preprocess(self):
        raise NotImplementedError

    def train(self):
        pass

    def render_plot(self):
        raise NotImplementedError

    def run(self, should_render_plot: bool = False):
        self.preprocess()
        self.train()

        if should_render_plot:
            self.render_plot()


if __name__ == "__main__":
    obj = SkeletonBasedActionRecognistion()
    obj.run()