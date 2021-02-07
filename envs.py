"""Environment Classes
"""
# Modules
import numpy as np

# Parameters

# Methods

# Classes
class MultiArmedBanditsAbstract:
    """Abstract Multi-Armed Bandit Class
    It should include the following functionalities:
    - reset : reset the environment
    - step : get action, update state and return reward
    """
    # Constructor
    def __init__( self, name, n_arms ):
        self.name = name
        self.n_arms = n_arms
        self.mus = None
        return
    # Reset
    def reset( self ):
        raise ValueError( '<reset> Not implemented.' )
        return
    # Step
    def step( self, action ):
        # Check Initialization Done
        assert not self.mus is None, "<step> Not initialized instance. Call 'reset' first"
        # Check Action Space
        assert isinstance(action, int) and 0 <= action < self.n_arms, "<step> action should be of type 'int' and in range: [0, {})".format(self.n_arms)
        # Reward
        reward = int( np.random.rand() <= self.mus[action] )
        # Return
        return( reward )

class MultiArmedBanditsDistribution( MultiArmedBanditsAbstract ):
    """Multi-Armed Bandits by Sampling Means from given Distribution
    """
    # Constructor
    def __init__( self, func_distribution, n_arms ):
        # Super
        super().__init__(
            name = "mab_distribution_{}_{}".format( n_arms, str(func_distribution) ),
            n_arms = n_arms
        )
        # Parameters
        self.func_distribution = func_distribution
        # Return
        return
    # Reset
    def reset( self ):
        self.mus = np.array([
            self.func_distribution() for _ in range(self.n_arms)
        ])
        return
class MultiArmedBanditsUniform( MultiArmedBanditsDistribution ):
    """Multi-Armed Bandits by Sampling Means from Uniform Distribution
    """
    # Constructor
    def __init__( self, n_arms ):
        # Super
        super().__init__( func_distribution = np.random.rand, n_arms = n_arms )
        # Return
        return

