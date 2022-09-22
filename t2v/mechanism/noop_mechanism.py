from PIL.Image import Image

from t2v.mechanism.mechanism import Mechanism


class NoopMechanism(Mechanism):
    """
    Mechanism that does nothing, for testing
    """
    def init(self, config):
        pass

    def generate(self, config, context, prompt):
        return Image.__init__('RGB')

    def destroy(self):
        pass

    @staticmethod
    def name():
        return "noop"
