class Mechanism:
    """
    Defines an arbitrary text2video mechanism.
    """

    def init(self, config):
        """
        Initialize the mechanism
        :param config: config parameters that configure the mechanism
        :return: None
        """
        pass

    def generate(self, config, context, prompt):
        """
        Generate a frame with the mechanism
        :param prompt: prompt to generate frames for
        :param config: configuration of the mechanism
        :param context: context from the previous generation, if any
        :return: the frame as a PIL image
        """

    def destroy(self):
        """
        de-initialize the mechanism. Unload models / free up memory as much as possible
        :return:
        """

    @staticmethod
    def name():
        """
        The name of the mechanism, by which the impl will be referenced in the config
        :return:
        """