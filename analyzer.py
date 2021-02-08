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
        self.mu_hat_actions = None
        self.mu_hats_max = None
        # Return
        return
    # Reset
    def reset( self ):
        self.actions = []
        self.rewards = []
        self.mu_hat_actions = []
        self.mu_hats_max = []
        self.env.reset()
        self.agent.reset()
        # Return
        return
    # Forward
    def forward( self, n_steps ):
        for i in range(n_steps):
            # Decide
            action = self.agent.decide()
            mu_hats = ( self.agent.rewards_sum + self.agent.reward_initial ) / ( self.agent.arm_pull_counts + 1 )
            mu_hat_action = mu_hats[action]
            mu_hats_max = mu_hats.max()
            # Get Reward
            reward = self.env.step( action )
            # Update Agent
            self.agent.update( action, reward )
            # Store
            self.actions.append( action )
            self.rewards.append( reward )
            self.mu_hat_actions.append( mu_hat_action )
            self.mu_hats_max.append( mu_hats_max )
        return
