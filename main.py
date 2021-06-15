from src.preprocessing.augmented_sig_transformer import AugmentedSigTransformer
from src.algos.sig_classifier import SigClassifier


class SkeletonBasedActionRecognistion:
    def __init__(self):
        self.sig_classifier = SigClassifier(
            sig_data_wrapper=AugmentedSigTransformer
        ) 

    def run(self, render_plot: bool = False):
        self.sig_classifier.run(render_plot=render_plot)

if __name__ == "__main__":
    obj = SkeletonBasedActionRecognistion()
    obj.run(render_plot=True)