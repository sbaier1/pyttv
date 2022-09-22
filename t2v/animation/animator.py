class Animator:
    """
    An Animator is an extension for a text2image model that can transform a frame as input for the next generation
    in order to induce motion
    """
    def init(self, config):
        """
        Initialize the mechanism, load models, etc.
        :return:
        """
    def apply(self, config, frame, prompt, context):
        """
        Apply the transformation of the animator to an input frame
        :param config:
        :param frame:
        :param prompt:
        :param context:
        :return:
        """
    def destroy(self, config):
        """
        Unload models, etc. and destroy this animator. It must not be reused after this method was invoked.
        :param config:
        :return:
        """