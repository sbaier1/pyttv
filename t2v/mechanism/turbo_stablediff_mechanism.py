from t2v.mechanism.mechanism import Mechanism


class TurboStableDiff(Mechanism):
    def init(self, config):
        super().init(config)

    def generate(self, config, context, prompt):
        super().generate(config, context, prompt)

    def destroy(self):
        super().destroy()

    @staticmethod
    def name():
        return "turbo-stablediff"
