"""Agent Classes
"""
# Modules
import numpy as np

# Parameters

# Methods

# Classes
class AgentAbstract:
    """Abstract class for agents
    """
    # Constructor
    def __init__( self, name, n_arms ):
        self.name = name
        self.n_arms = n_arms
        self.reset()
        return
    # Reset
    def reset( self ):
        self.rewards_sum = np.zeros( n_arms )
        self.arm_pull_counts = np.zeros( n_arms, dtype=np.int )
        self.action_weights = np.ones( n_arms )
        self.actions = np.arange( self.n_arms )
        return
    # Decide
    def decide( self ):
        action = np.random.choice( self.actions, p = self.action_weights / self.action_weights.sum() )
        return( action )
    # Update
    def update( self, action, reward ):
        # Check Action Space
        assert isinstance(action, int) and 0 <= action < self.n_arms, "<update> action should be of type 'int' and in range: [0, {})".format(self.n_arms)
        # General Updates
        self.arm_pull_counts[action] += 1
        self.rewards_sum[action] += reward
        # Model Specific Computations
        self._update_model_specific( action, reward )
        # Calculate Weights
        self._calculate_weights()
        # Return
        return
    # Update Model Specific
    def _update_model_specific( self ):
        raise ValueError( '<_update_model_specific> Not implemented.' )
        return
    # Calculate Weights
    def _calculate_weights( self ):
        raise ValueError( '<_calculate_weights> Not implemented.' )
        return
class AgentGreedy( AgentAbstract ):
    """Greedy Agent
    """
    # Constructor
    def __init__( self, n_arms, reward_initial = 1.0 ):
        # Super
        super().__init__( name = "agent_greedy_{}_{}".format( n_arms, reward_initial ), n_arms = n_arms )
        # Parameters
        self.reward_initial = reward_initial
        self._calculate_weights()
        # Return
        return
    # Update Model Specific
    def _update_model_specific( self, action, reward ):
        return
    # Calculate Weights
    def _calculate_weights( self ):
        self.action_weights = ( self.rewards_sum + self.reward_initial ) / ( self.arm_pull_counts + 1 )
        return
class AgentGreedyEps( AgentGreedy ):
    """Greedy Agent + Epsilon Exploration
    """
    # Constructor
    def __init__( self, n_arms, epsilon, reward_initial = 1.0 ):
        # Super
        super().__init__( n_arms = n_arms, reward_initial = reward_initial )
        self.name = "agent_greedy_epsilon_{}_{}_{}".format( n_arms, epsilon, reward_initial )
        # Parameters
        self.epsilon = epsilon
        self._calculate_weights()
        # Return
        return
    # Decide
    def decide( self ):
        if( np.random.rand() <= self.epsilon ):
            action = np.random.choice( self.actions )
        else:
            action = super().decide()
        return( action )
