"""
============
Polarization
============

Provides functions for:
- Creating initial belief configurations for various scenarios.
- Creating influence graphs for various scenarios.
- The Esteban-Ray polarization measure.
- Discretizing a belief state into a distribution.
- Updating the belief state of an agent, see the `Update` enum for all types.

"""

import math
from enum import Enum
from functools import partial

import numpy as np

######################################################
## Parameters for the Belief states
######################################################

## number of agents
NUM_AGENTS = 100

## for consensus belief-function: belief value for the consensus belief state
CONSENSUS_VALUE = 0.5

## Values for the old belief configurations:
## -----------------------------------------
## for mildly-polarized belief-function: belief value for the upper end of the low pole of a mildly polarized belief state
LOW_POLE = 0.25
## for mildly-polarized belief-function: belief value for the lower end of the high pole of a mildly polarized belief state
HIGH_POLE = 0.75
## for mildly-polarized belief-function: step of belief change from agent to agent in mildly polarized belief state
BELIEF_STEP = 0.01

######################################################
## Parameters for the Esteban-Ray polarization measure
######################################################

## number of bins when discretizing the belief state
NUM_BINS = 5
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

#######################################
## Parameters for influence graphs
#######################################

## for clique influence-graph: belief value of all agents on a clique influence graph
CLIQUE_INF_VALUE = 0.5

## for 2_groups_disconnected influence-graph: belief value of all agents that can communicate in a 2 groups_disconnected influence graph
GROUPS_DISCONNECTED_INF_VALUE = 0.5

## for 2_groups_faint ineraction-function: belief value of all agents that can strongly communicate in a 2 groups faintly connected influence graph
GROUPS_FAINTLY_INF_VALUE_STRONG = 0.5
## for 2_groups_faint ineraction-function: belief value of all agents that can weakly communicate in a 2 groups faintly connected influence graph
GROUPS_FAINTLY_INF_VALUE_WEAK = 0.1

## for 2_influencers_balanced influence-graph: level of influence both influencers exert on all others
INFLUENCERS_BALANCED_OUTGOING_BOTH = 0.6
## for 2_influencers_balanced influence-graph: level of influence both influencers receive from all others
INFLUENCERS_BALANCED_INCOMING_BOTH = 0.0
## for 2_influencers_balanced influence-graph: level of influence all other agents exert on all others
INFLUENCERS_BALANCED_OTHERS = 0.1

## for 2_influencers_unbalanced influence-graph: level of influence agent 0 exerts on all others
INFLUENCERS_UNBALANCED_OUTGOING_FIRST = 0.8
## for 2_influencers_unbalanced influence-graph: level of influence agent n-1 exerts on all others
INFLUENCERS_UNBALANCED_OUTGOING_SECOND = 0.5
## for 2_influencers_unbalanced influence-graph: level of influence agent 0 receives from all others
INFLUENCERS_UNBALANCED_INCOMING_FIRST = 0.1
## for 2_influencers_unbalanced influence-graph: level of influence agent n-1 receives from all others
INFLUENCERS_UNBALANCED_INCOMING_SECOND = 0.1
## for 2_influencers_unbalanced influence-graph: level of influence all other agents exert on all others
INFLUENCERS_UNBALANCED_OTHERS = 0.2

## for circular influence-graph: belief value of all agents on a circular influence graph
CIRCULAR_INF_VALUE = 0.5

############################################
## Representing belief states implementation
############################################

class Belief(Enum):
    UNIFORM = 0
    MILD = 1
    EXTREME = 2
    TRIPLE = 3
    CONSENSUS = 4

## Current representation

def build_belief(belief_type: Belief, num_agents=NUM_AGENTS, **kwargs):
    """Evenly distributes the agents beliefs into subgroups.

    Same inputs as `build_old_belief`, will call it in case of finding a non-defined case
    The default values are the constants defined at the beginning of the polarization module.
    """
    if belief_type is Belief.MILD:
        middle = math.ceil(num_agents / 2)
        return [0.2 + 0.2 * i / middle if i < middle else 0.6 + 0.2 * (i - middle) / (num_agents - middle) for i in range(num_agents)]
    if belief_type is Belief.EXTREME:
        middle = math.ceil(num_agents / 2)
        return [0.2 * i / middle if i < middle else 0.8 + 0.2 * (i - middle) / (num_agents - middle) for i in range(num_agents)]
    if belief_type is Belief.TRIPLE:
        beliefs = [0.0] * num_agents
        first_third = num_agents // 3
        middle_third = math.ceil(num_agents * 2 / 3) - first_third
        last_third = num_agents - middle_third - first_third
        offset = 0
        for i, segment in enumerate((first_third, middle_third, last_third)):
            for j in range(segment):
                beliefs[j+offset] = 0.2 * j / segment + (0.4 * i)
            offset += segment
        return beliefs
    return build_old_belief(belief_type, num_agents, **kwargs)

## Old representation

def build_uniform_beliefs(num_agents):
    """ Build uniform belief state.
    """
    return [i/(num_agents - 1) for i in range(num_agents)]

def build_old_mild_beliefs(num_agents, low_pole, high_pole, step):
    """Builds mildly polarized belief state, in which 
    half of agents has belief decreasing from 0.25, and
    half has belief increasing from 0.75, all by the given step.
    """
    middle = math.ceil(num_agents / 2)
    return [max(low_pole - step*(middle - i - 1), 0) if i < middle else min(high_pole - step*(middle - i), 1) for i in range(num_agents)]

def build_old_extreme_beliefs(num_agents):
    """Builds extreme polarized belief state, in which half
    of the agents has belief 0, and half has belief 1.
    """
    middle = math.ceil(num_agents / 2)
    return [0 if i < middle else 1 for i in range(num_agents)]

def build_old_triple_beliefs(num_agents):
    """Builds three-pole belief state, in which each 
    one third of the agents has belief 0, one third has belief 0.4,
    and one third has belief 1.
    """
    beliefs = [0.0] * num_agents
    one_third = num_agents // 3
    two_thirds = math.ceil(2 * num_agents / 3)
    for i in range(num_agents):
        if i >= two_thirds:
            beliefs[i] = 1.0
        elif i >= one_third:
            beliefs[i] = 0.5
    return beliefs

def build_consensus_beliefs(num_agents, belief):
    """Builds consensus belief state, in which each 
    all agents have same belief.
    """
    return [belief] * num_agents

def build_old_belief(
        belief_type,
        num_agents=NUM_AGENTS,
        *,
        low_pole=LOW_POLE,
        high_pole=HIGH_POLE,
        step=BELIEF_STEP,
        consensus_value=CONSENSUS_VALUE):
    """Builds the initial belief state according to the `belief_type`.

    Helper function when iterating over the `Belief` enum. The default values
    are the constants defined at the beginning of the polarization module.
    """
    if belief_type is Belief.UNIFORM:
        return build_uniform_beliefs(num_agents)
    if belief_type is Belief.MILD:
        return build_old_mild_beliefs(num_agents, low_pole, high_pole, step)
    if belief_type is Belief.EXTREME:
        return build_old_extreme_beliefs(num_agents)
    if belief_type is Belief.TRIPLE:
        return build_old_triple_beliefs(num_agents)
    if belief_type is Belief.CONSENSUS:
        return build_consensus_beliefs(num_agents, consensus_value)
    raise Exception('belief_type not recognized. Expected a `Belief`')

######################################################
## The Esteban-Ray polarization measure implementation
######################################################

def belief_2_dist(belief_vec, num_bins=NUM_BINS):
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

def make_belief_2_dist_func(num_bins=NUM_BINS):
    """Returns a function that discretizes a belief state into a `num_bins`
    number of bins.
    """
    _belief_2_dist = partial(belief_2_dist, num_bins=num_bins)
    _belief_2_dist.__name__ = belief_2_dist.__name__
    _belief_2_dist.__doc__ = belief_2_dist.__doc__
    return _belief_2_dist

def pol_ER(dist, alpha=ALPHA, K=K):
    """Computes the Esteban-Ray polarization of a distribution.
    """
    # recover bin labels
    bin_labels = dist[0]
    # recover bin probabilities
    bin_prob = dist[1]

    diff = np.ones((len(bin_labels), 1)) @ bin_labels[np.newaxis]
    diff = np.abs(diff - np.transpose(diff))
    pol = (bin_prob ** (1 + alpha)) @ diff @ bin_prob
    # scales the measure by the constant K, and returns it.
    return K * pol

def make_pol_er_func(alpha=ALPHA, K=K):
    """Returns a function that computes the Esteban-Ray polarization of a
    distribution with set parameters `alpha` and `K`
    """
    _pol_ER = partial(pol_ER, alpha=alpha, K=K)
    _pol_ER.__name__ = pol_ER.__name__
    _pol_ER.__doc__ = pol_ER.__doc__
    return _pol_ER

def pol_ER_discretized(belief_state, alpha=ALPHA, K=K, num_bins=NUM_BINS):
    """Discretize belief state as necessary for computing Esteban-Ray
    polarization and computes it.
    """
    return pol_ER(belief_2_dist(belief_state, num_bins), alpha, K)

def make_pol_er_discretized_func(alpha=ALPHA, K=K, num_bins=NUM_BINS):
    """Returns a function that computes the Esteban-Ray polarization of a
    belief state, discretized into a `num_bins` number of bins, with set
    parameters `alpha` and `K`.
    """
    _pol_ER_discretized = partial(pol_ER_discretized, alpha=alpha, K=K, num_bins=num_bins)
    _pol_ER_discretized.__name__ = pol_ER_discretized.__name__
    _pol_ER_discretized.__doc__ = pol_ER_discretized.__doc__
    return _pol_ER_discretized

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

def build_inf_graph_circular(num_agents, value):
    """Returns the imfluemce graph for "circular influence" scenario."""
    inf_graph = np.zeros((num_agents, num_agents))
    for i in range(num_agents):
        inf_graph[i, i] = 1.0
        inf_graph[i, (i+1) % num_agents] = value
    return inf_graph

class Influence(Enum):
    CLIQUE = 0
    GROUP_2_DISCONECTED = 1
    GROUP_2_FAINT = 2
    INFLUENCERS_2_BALANCED = 3
    INFLUENCERS_2_UNBALANCED = 4
    CIRCULAR = 5

def build_influence(
        inf_type: Influence,
        num_agents=NUM_AGENTS,
        *,
        weak_belief=GROUPS_FAINTLY_INF_VALUE_WEAK,
        strong_belief=GROUPS_FAINTLY_INF_VALUE_STRONG,
        general_belief=None,
        influencer_incoming_belief=None,
        influencer_outgoing_belief=None,
        influencer2_incoming_belief=INFLUENCERS_UNBALANCED_INCOMING_SECOND,
        influencer2_outgoing_belief=INFLUENCERS_UNBALANCED_OUTGOING_SECOND):
    """Builds the initial influence graph according to the `inf_type`.

    Helper function when iterating over the `Influence` enum. The default values
    are the constants defined at the beginning of the polarization module.
    """
    if inf_type is Influence.CLIQUE:
        if general_belief is None:
            general_belief = CLIQUE_INF_VALUE
        return build_inf_graph_clique(num_agents, general_belief)
    if inf_type is Influence.GROUP_2_DISCONECTED:
        if general_belief is None:
            general_belief = GROUPS_DISCONNECTED_INF_VALUE
        return build_inf_graph_2_groups_disconnected(num_agents, general_belief)
    if inf_type is Influence.GROUP_2_FAINT:
        return build_inf_graph_2_groups_faint(num_agents, weak_belief, strong_belief)
    if inf_type is Influence.INFLUENCERS_2_BALANCED:
        if general_belief is None:
            general_belief = INFLUENCERS_BALANCED_OTHERS
        if influencer_incoming_belief is None:
            influencer_incoming_belief = INFLUENCERS_BALANCED_INCOMING_BOTH
        if influencer_outgoing_belief is None:
            influencer_outgoing_belief = INFLUENCERS_BALANCED_OUTGOING_BOTH
        return build_inf_graph_2_influencers_balanced(num_agents, influencer_incoming_belief, influencer_outgoing_belief, general_belief)
    if inf_type is Influence.INFLUENCERS_2_UNBALANCED:
        if general_belief is None:
            general_belief = INFLUENCERS_UNBALANCED_OTHERS
        if influencer_incoming_belief is None:
            influencer_incoming_belief = INFLUENCERS_UNBALANCED_INCOMING_FIRST
        if influencer2_incoming_belief is None:
            influencer2_incoming_belief = INFLUENCERS_UNBALANCED_INCOMING_SECOND
        if influencer_outgoing_belief is None:
            influencer_outgoing_belief = INFLUENCERS_UNBALANCED_OUTGOING_FIRST
        if influencer2_outgoing_belief is None:
            influencer2_outgoing_belief = INFLUENCERS_UNBALANCED_OUTGOING_SECOND
        return build_inf_graph_2_influencers_unbalanced(num_agents, influencer_outgoing_belief, influencer2_outgoing_belief, influencer_incoming_belief, influencer2_incoming_belief, general_belief)
    if inf_type is Influence.CIRCULAR:
        if general_belief is None:
            general_belief = CIRCULAR_INF_VALUE
        return build_inf_graph_circular(num_agents, general_belief) 
    raise Exception('inf_type not recognized. Expected an `Influence`')

#####################################
## Update Functions Implementation
#####################################

class Update(Enum):
    CLASSIC = 1
    CONFBIAS = 2

def neighbours_update(beliefs, inf_graph):
    """Applies the classic update function as matrix multiplication.
    
    For each agent, update their beliefs factoring the authority bias and
    the beliefs of all the agents' neighbors.

    Equivalent to:

    [blf_ai + np.mean([inf_graph[other, agent] * (blf_aj - blf_ai) for other, blf_aj in enumerate(beliefs) if inf_graph[other, agent] > 0]) for agent, blf_ai in enumerate(beliefs)]

    """
    neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
    return (beliefs @ inf_graph - np.add.reduce(inf_graph) * beliefs) / neighbours + beliefs

def neighbours_cb_update(beliefs, inf_graph):
    """Applies the confirmation-bias update function.
    
    For each agent, update their beliefs factoring the authority bias,
    confirmation-bias and the beliefs of all the agents' neighbors.
    """
    return [blf_ai + np.mean([(1 - np.abs(blf_aj - blf_ai)) * inf_graph[other, agent] * (blf_aj - blf_ai) for other, blf_aj in enumerate(beliefs) if inf_graph[other, agent] > 0 or other == agent]) for agent, blf_ai in enumerate(beliefs)]

def np_neighbours_cb_update(beliefs, inf_graph):
    """Applies the classic update function as matrix multiplication.
    
    For each agent, update their beliefs factoring the authority bias,
    confirmation-bias and the beliefs of all the agents' neighbors.

    Equivalent to:
    [blf_ai + np.mean([(1 - np.abs(blf_aj - blf_ai)) * inf_graph[other, agent] * (blf_aj - blf_ai) for other, blf_aj in enumerate(beliefs) if inf_graph[other, agent] > 0]) for agent, blf_ai in enumerate(beliefs)]
    """
    neighbours = [np.count_nonzero(inf_graph[:, i]) for i, _ in enumerate(beliefs)]
    diff = np.ones((len(beliefs), 1)) @ beliefs[np.newaxis]
    diff = np.transpose(diff) - diff
    infs = inf_graph * (1 - np.abs(diff)) * diff
    return np.add.reduce(infs) / neighbours + beliefs

def normalize_graph(inf_graph):
    """Allows us to use apply the DeGroot Update function."""
    num_agents = len(inf_graph)
    result = inf_graph / num_agents
    for agent in range(num_agents):
        result[agent, agent] = 1 - sum(inf[agent] for pos, inf in enumerate(result) if pos != agent)
    return result

def degroot_update(beliefs, inf_graph):
    """Applies the DeGroot Update function."""
    infs = normalize_graph(inf_graph)
    # TODO: replace with matrix multiplication
    return [sum(blf * infs[other, agent] for other, blf in enumerate(beliefs)) for agent, _ in enumerate(beliefs)]

def make_update_fn(update_type: Update):
    if update_type is Update.CLASSIC:
        return neighbours_update
    if update_type is Update.CONFBIAS:
        return neighbours_cb_update
    if isinstance(update_type, Update):
        raise NotImplementedError(f"Not implemented for {update_type}")
    else:
        raise TypeError(f"update_type must be an Update")

######################################
## Old Update Functions Implementation
######################################

class OldUpdate(Enum):
    CLASSIC = 1
    CONFBIAS_SMOOTH = 2
    CONFBIAS_SHARP = 3
    BACKFIRE = 4

def update_all(belief_vec, inf_graph, update_type: OldUpdate, confbias_discount, backfire_belief_threshold, backfire_influence_threshold):
    """Updates the whole belief state"""
    num_agents = len(belief_vec)
    return [update_agent_vs_all(belief_vec, inf_graph, agent, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold) for agent in range(num_agents)]

def update_all_np(belief_vec, inf_graph, update_type: OldUpdate, confbias_discount, backfire_belief_threshold, backfire_influence_threshold):
    """Updates the whole belief state"""
    num_agents = len(belief_vec)
    return [np.mean([update_agent_pair(belief_vec[agent], belief_vec[other], inf_graph[other, agent], update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold) for other in range(num_agents)]) for agent in range(num_agents)]

def update_agent_vs_all(belief_vec, inf_graph, agent, update_type: OldUpdate, confbias_discount, backfire_belief_threshold, backfire_influence_threshold):
    """Updates the belief value of one individual agent considering the effect
    of all other agents on him.
    """
    num_agents = len(belief_vec)
    belief = sum((update_agent_pair(belief_vec[agent], belief_vec[other], inf_graph[other, agent], update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold) for other in range(num_agents)))
    return belief / num_agents

def update_agent_pair(belief_ai, belief_aj, influence, update_type: OldUpdate, confbias_discount, backfire_belief_threshold, backfire_influence_threshold):
    """Updates the belief value of one individual ai considering the effect one
    other individual aj on him.
    """
    if update_type is OldUpdate.CLASSIC:
        return belief_ai + influence * (belief_aj - belief_ai)
    elif update_type is OldUpdate.CONFBIAS_SMOOTH:
        confbias_factor = 1 - np.abs(belief_aj - belief_ai)
        return belief_ai + confbias_factor * influence * (belief_aj - belief_ai)
    elif update_type is OldUpdate.CONFBIAS_SHARP:
        if (belief_aj - 0.5) * (belief_ai - 0.5) >= 0:
            confbias_factor = 1
        else:
            confbias_factor = confbias_discount
        return belief_ai + confbias_factor * influence * (belief_aj - belief_ai)
    elif update_type is OldUpdate.BACKFIRE:
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

def run_until_stable(belief_vec, inf_graph, max_time=100, num_bins=NUM_BINS, update_type=OldUpdate.CLASSIC, confbias_discount=CONFBIAS_DISCOUNT, backfire_belief_threshold=BACKFIRE_BELIEF_THRESHOLD, backfire_influence_threshold=BACKFIRE_INFLUENCE_THRESHOLD):
    pol_ER_discretized = make_pol_er_discretized_func(ALPHA, K, num_bins)

    ## Creates temporary values to store evolution with time.
    belief_vec_state = belief_vec
    pol_state = pol_ER_discretized(belief_vec_state)

    belief_history = [belief_vec_state]
    pol_history = [pol_state]

    ## Execites simulation for max_time steps or convergence.
    for _ in range(1, max_time):
        # Update beliefs
        belief_vec_state = update_all_np(belief_vec_state, inf_graph, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold)
        # Compute Esteban-Ray polarization.
        pol_state = pol_ER_discretized(belief_vec_state)
        # Appends the new polarization state to the log.
        pol_history.append(pol_state)
        belief_history.append(belief_vec_state)

        if belief_history[-2] == belief_history[-1]:
            break

    return (np.array(pol_history), np.array(belief_history), pol_history[-1])

def make_old_update_fn(update_type: OldUpdate, confbias_discount=CONFBIAS_DISCOUNT, backfire_belief_threshold=BACKFIRE_BELIEF_THRESHOLD, backfire_influence_threshold=BACKFIRE_INFLUENCE_THRESHOLD):
    def update_fn(belief_vec, inf_graph):
        return update_all(belief_vec, inf_graph, update_type, confbias_discount, backfire_belief_threshold, backfire_belief_threshold)
    return update_fn

#######################################
## Main Simulation Class Implementation
#######################################

class Simulation:
    def __init__(self, belief_vec, inf_graph, update_fn=make_update_fn(Update.CLASSIC), pol_measure=None, num_bins=NUM_BINS):
        self.belief_vec = np.array(belief_vec)
        self.inf_graph = inf_graph
        self.update_fn = update_fn
        self.num_bins = num_bins
        self.pol_measure = pol_measure

    def __iter__(self):
        return self

    def __next__(self):
        if self.pol_measure is None:
            result = (self.belief_vec, pol_ER_discretized(self.belief_vec, num_bins=self.num_bins))
        else:
            result = (self.belief_vec, self.pol_measure(self.belief_vec))
        self.belief_vec = self.update_fn(self.belief_vec, self.inf_graph)
        return result

    def run(self, max_time=100, smart_stop=True):
        """Runs the current Simulation setup.

        For each time step, calculate the polarization value and the belief
        state. If `smart_stop` is `True`, the simulation stops when the
        belief state does not change from one time step to the next or if
        `max_time` is reached.

        Args:
          - max_time (int, default 100): The time step to stop the simulation.
          - smart_stop (Boolean, default True): Stops the simulation if the
            belief state stabilizes or `max_time` is reached.

        Returns:
          A tuple of NumPy Arrays with each polarization value, its
          corresponding  belief state, and the last polarization value.
          (pol_history, blf_history, pol_history[-1]).
        """
        belief_history = []
        pol_history = []
        for _, (belief_vec_state, pol_state) in zip(range(max_time), self):
            # Stop if a stable state is reached
            if smart_stop and belief_history and np.array_equal(belief_history[-1], belief_vec_state):
                break
            belief_history.append(belief_vec_state)
            pol_history.append(pol_state)
        return (np.array(pol_history), np.array(belief_history), pol_history[-1])
