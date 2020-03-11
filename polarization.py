"""
============
Polarization
============

Provides functions for:
- Creating initial belief states for various scenarios.
- Creating influence graphs for various scenarios.
- The Esteban-Ray polarization measure.
- Discretizing a belief state into a distribution.
- Updating the belief state of an agent, see the `Update` enum for all types.

"""

import math
from enum import Enum

import numpy as np

######################################################
## Parameters for the Belief states
######################################################

## number of agents
NUM_AGENTS = 100

## for mildly-polarized belief-function: belief value for the upper end of the low pole of a mildly polarized belief state
LOW_POLE = 0.25
## for mildly-polarized belief-function: belief value for the lower end of the high pole of a mildly polarized belief state
HIGH_POLE = 0.75
## for mildly-polarized belief-function: step of belief change from agent to agent in mildly polarized belief state
BELIEF_STEP = 0.01

## for consensus belief-function: belief value for the consensus belief state
CONSENSUS_VALUE = 0.5

######################################################
## Parameters for the Esteban-Ray polarization measure
######################################################

## number of bins when discretizing the belief state
NUM_BINS = 201
## parameter alpha set to what Esteban and Ray suggest
ALPHA = 1.6
## scaling factor for polarization measure
K = 1000

##################################
## Parameters for update functions
##################################

## for conf_bias update-function: confirmation bias discount
CONFBIAS_DISCOUNT = 0.5
## for backfire_effect update-function: minimum difference of beliefs to trigger the backfire effect
BACKFIRE_BELIEF_THRESHOLD = 0.4
## for backfire_effect update-function: minimum influence to trigger the backfire effect
BACKFIRE_INFLUENCE_THRESHOLD = 0.2

############################################
## Representing belief states implementation
############################################

def build_uniform_beliefs(num_agents):
    """ Build uniform belief state.
    """
    return [i/(num_agents - 1) for i in range(num_agents)]

def build_mild_beliefs(num_agents, low_pole, high_pole, step):
    """Builds mildly polarized belief state, in which 
    half of agents has belief decreasing from 0.25, and
    half has belief increasing from 0.75, all by the given step.
    """
    middle = num_agents // 2
    return [max(low_pole - step*(middle - i), 0) if i <= middle else min(high_pole - step*(middle - i + 1), 1) for i in range(num_agents)]

def build_extreme_beliefs(num_agents):
    """Builds extreme polarized belief state, in which half
    of the agents has belief 0, and half has belief 1.
    """
    middle = num_agents // 2
    return [0 if i <= middle else 1 for i in range(num_agents)]

def build_triple_beliefs(num_agents):
    """Builds three-pole belief state, in which each 
    one third of the agents has belief 0, one third has belief 0.4,
    and one third has belief 1.
    """
    beliefs = [0.0] * num_agents
    one_third = num_agents // 3
    two_thirds = 2 * num_agents // 3
    for i in range(num_agents):
        if i > two_thirds:
            beliefs[i] = 1.0
        elif i >= one_third:
            beliefs[i] = 0.5
    return beliefs

def build_consensus_beliefs(num_agents, belief):
    """Builds consensus belief state, in which each 
    all agents have same belief.
    """
    return [belief] * num_agents

######################################################
## The Esteban-Ray polarization measure implementation
######################################################

def make_belief_2_dist_func(num_bins):
    """Returns a function that discretizes a belief state into a `num_bins`
    number of bins.
    """
    def belief_2_dist(belief_vec):
        """Takes a belief state `belief_vec` and discretizes it into a fixed
        number of bins.
        """
        # stores labels of bins
        # the value of a bin is the medium point of that bin
        bin_labels = [(i + 0.5)/num_bins for i in range(num_bins)]

        # stores the distribution of labels
        bin_prob = [0] * num_bins
        # for all agents...
        for belief in belief_vec:
            # computes the bin into which the agent's belief falls
            bin_ = math.floor(belief * num_bins)
            # treats the extreme case in which belief is 1, putting the result in the right bin.
            if bin_ == num_bins:
                bin_ = num_bins - 1
            # updates the frequency of that particular belief
            bin_prob[bin_] += 1 / len(belief_vec)
        # bundles into a matrix the bin_labels and bin_probabilities.
        dist = np.array([bin_labels,bin_prob])
        # returns the distribution.
        return dist
    return belief_2_dist

def make_pol_er_func(alpha, K):
    """Returns a function that computes the Esteban-Ray polarization of a
    distribution with set parameters `alpha` and `K`
    """
    def pol_ER(dist):
        """Computes the Esteban-Ray polarization of a distribution.
        """
        # recover bin labels
        bin_labels = dist[0]
        # recover bin probabilities
        bin_prob = dist[1]
        # number of bins
        num_bins = len(bin_labels)
        # initialize polarization measure
        pol = 0
        # computes the esteban-ray measure
        for i in range(num_bins):
            for j in range(num_bins):
                pol += ((bin_prob[i]**(1 + alpha)) * bin_prob[j]) * abs(bin_labels[i] - bin_labels[j]) 
        # scales the measure by the constant K, and returns it.
        return K * pol
    return pol_ER

def make_pol_er_discretized_func(alpha, K, num_bins):
    """Returns a function that computes the Esteban-Ray polarization of a
    belief state, discretized into a `num_bins` number of bins, with set
    parameters `alpha` and `K`.
    """
    belief_2_dist = make_belief_2_dist_func(num_bins)
    pol_ER = make_pol_er_func(ALPHA, K)
    def pol_ER_discretized(belief_state):
        """Discretize belief state as necessary for computing Esteban-Ray
        polarization and computes it.
        """
        return pol_ER(belief_2_dist(belief_state))
    return pol_ER_discretized

#####################################
## Influence graphs implementation
#####################################

def build_inf_graph_clique(num_agents, belief_value):
    """Returns the influence graph for "clique" scenario."""
    return np.full((num_agents, num_agents), belief_value)

def build_inf_graph_2_groups_disconnected(num_agents, belief_value):
    """Returns the influence graph for for "disconnected" scenario."""
    inf_graph = np.zeros((num_agents, num_agents))
    middle = math.ceil(num_agents / 2)
    inf_graph[:middle, :middle] = belief_value
    inf_graph[middle:, middle:] = belief_value
    return inf_graph

def build_inf_graph_2_groups_faint(num_agents, weak_belief_value, strong_belief_value):
    """Returns the influence graph for for "faintly-connected" scenario."""
    inf_graph = np.full((num_agents, num_agents), weak_belief_value)
    middle = math.ceil(num_agents / 2)
    inf_graph[:middle, :middle] = strong_belief_value
    inf_graph[middle:, middle:] = strong_belief_value
    return inf_graph
  
def build_inf_graph_2_influencers_balanced(num_agents, influencers_incoming_value, influencers_outgoing_value, others_belief_value):
    """Returns the influence graph for for "balanced 2-influencers" scenario."""
    inf_graph = np.full((num_agents, num_agents), others_belief_value)
    ## Sets the influence of agent 0 on all others
    inf_graph[0, :-1] = influencers_outgoing_value      
    ## Sets the influence of agent n-1 on all others
    inf_graph[-1, 1:] = influencers_outgoing_value
    ## Sets the influence of all other agents on agent 0.
    inf_graph[1:,0] = influencers_incoming_value
    ## Sets the influence of all other agents on agent n-1.
    inf_graph[:-1, -1] = influencers_incoming_value    
    return inf_graph

def build_inf_graph_2_influencers_unbalanced(num_agents, influencers_outgoing_value_first, influencers_outgoing_value_second, influencers_incoming_value_first, influencers_incoming_value_second, others_belief_value):
    """Returns the influence graph for for "unbalanced 2-influencers" scenario."""
    inf_graph = np.full((num_agents,num_agents), others_belief_value)
    ## Sets the influence of agent 0 on all others
    inf_graph[0, :-1] = influencers_outgoing_value_first       
    ## Sets the influence of agent n-1 on all others
    inf_graph[-1, 1:] = influencers_outgoing_value_second
    ## Sets the influence of all other agents on agent 0.
    inf_graph[1:, 0] = influencers_incoming_value_first
    ## Sets the influence of all other agents on agent n-1.
    inf_graph[:-1, -1] = influencers_incoming_value_second
    return inf_graph

#####################################
## Update Functions Implementation
#####################################

class Update(Enum):
    CLASSIC = 1
    CONFBIAS_SMOOTH = 2
    CONFBIAS_SHARP = 3
    BACKFIRE = 4

def update_all(belief_vec, inf_graph, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold):
    """Updates the whole belief state"""
    num_agents = len(belief_vec)
    return [update_agent_vs_all(belief_vec, inf_graph, agent, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold) for agent in range(num_agents)]

def update_agent_vs_all(belief_vec, inf_graph, agent, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold):
    """Updates the belief value of one individual agent considering the effect
    of all other agents on him.
    """
    num_agents = len(belief_vec)
    belief = sum((update_agent_pair(belief_vec[agent], belief_vec[other], inf_graph[other, agent], update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold) for other in range(num_agents)))
    return belief / num_agents

def update_agent_pair(belief_ai, belief_aj, influence, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold):
    """Updates the belief value of one individual ai considering the effect one
    other individual aj on him.
    """
    if update_type is Update.CLASSIC:
        return belief_ai + influence * (belief_aj - belief_ai)
    elif update_type is Update.CONFBIAS_SMOOTH:
        confbias_factor = 1 - np.abs(belief_aj - belief_ai)
        return belief_ai + confbias_factor * influence * (belief_aj - belief_ai)
    elif update_type is Update.CONFBIAS_SHARP:
        if (belief_aj - 0.5) * (belief_ai - 0.5) >= 0:
            confbias_factor = 1
        else:
            confbias_factor = confbias_discount
        return belief_ai + confbias_factor * influence * (belief_aj - belief_ai)
    elif update_type is Update.BACKFIRE:
        # Compute the absolute difference in belief between agents
        deltaij = np.abs(belief_ai - belief_aj)

        # Check if backfire is to be triggered
        if deltaij >= backfire_belief_threshold and influence >= backfire_influence_threshold:
            backfire_factor = deltaij / influence
            
            # This transformation takes the step function from (-1, 1) to (0, 1)
            z = (2 * belief_ai) - 1
            
            # Compute the backfire update
            new_belief = 1 / (1 + math.e ** (-2 * backfire_factor * z))
            if belief_ai <= 0.5:
                new_belief = min(belief_ai, new_belief)
            else:
                new_belief = max(belief_ai, new_belief)
        else:
            new_belief = belief_ai + influence * (belief_aj - belief_ai)
        
        return new_belief
    # If the update function is not defined
    return 0

def run_simulation(belief_vec, inf_graph, max_time, num_bins, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold):
    ## Initialize functions
    ## TODO: Pass them as arguments, use and adapter to pol_er
    belief_2_dist = make_belief_2_dist_func(num_bins)
    pol_ER = make_pol_er_func(ALPHA, K)
    def pol_ER_discretized(belief_state):
        """Discretize belief state as necessary for computing Esteban-Ray
        polarization and computes it.
        """
        return pol_ER(belief_2_dist(belief_state))
    
    ## Creates temporary values to store evolution with time.
    belief_vec_state = belief_vec
    pol_state = pol_ER_discretized(belief_vec_state)

    pol_history = [pol_state]

    ## Execites simulation for max_time steps.
    for _ in range(1, max_time):
        # Update beliefs
        belief_vec_state = update_all(belief_vec_state, inf_graph, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold)
        # Compute Esteban-Ray polarization.
        pol_state = pol_ER_discretized(belief_vec_state)
        # Appends the new polarization state to the log.
        pol_history.append(pol_state)

    return np.array(pol_history)

