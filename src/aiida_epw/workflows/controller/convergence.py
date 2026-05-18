
class ConvergenceController():
    """ This class controls the convergence of the EpwPrepWorkChain.
    """
    def __init__(self, grp_name):
        self.convergence_type = 'bands'
        self.grp_name = grp_name

    def get_builder(self):
        """ Return the builder for the next step of the workflow. """
        pass