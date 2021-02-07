"""Analyzer Module to Analyze Single Agent - Environment Interactions
"""
# Modules

# Parameters

# Methods

# Classes
class Analyzer:
    """Analyzer Class to Simulate Agent-Environment Interactions
    """
    # Constructor
    def __init__( self, env_instance, agent_instance ):
        # Parameters
        self.env = env_instance
        self.agent = agent_instance
        self.actions = None
        self.rewards = None
        self.action_weights = None
        self.action_weights_max = None
        # Return
        return
    # Reset
    def reset( self ):
        self.actions = []
        self.rewards = []
        self.action_weights = []
        self.action_weights_max = []
        self.env.reset()
        self.agent.reset()
        # Return
        return
    # Continue
    def continue( self, n_steps ):
        for i in range(n_steps):
            # Decide
            action = self.agent.decide()
            action_weight = self.agent.action_weights[action]
            actions_weight_max = self.agent.action_weights.max()
            # Get Reward
            reward = self.env.step( action )
            # Store
            self.actions.append( action )
            self.rewards.append( reward )
            self.action_weights.append( action_weight )
            self.action_weights_max.append( actions_weight_max )
        return
