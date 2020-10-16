import polarization
import numpy as np

###################
# Basic Unit Tests
###################

def test_equality(expected, actual, name):
    message = "test {} ...".format(name)
    if expected != actual:
        print("{} \x1b[31mFAILED\x1b[0m".format(message))
        print("Expected:", expected)
        print("  Actual:", actual)
        return False
    else:
        print("{} \x1b[32mok\x1b[0m".format(message))
        return True

def test_build_uniform_beliefs():
    num_agents = 5
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]

    t1 = test_function(polarization.build_uniform_beliefs, expected, (num_agents,))

    return (t1,)

def test_build_mild_beliefs():
    num_agents = 5
    expected = [0.19999999999999998, 0.25, 0.3, 0.7, 0.75]
    
    t1 = test_function(polarization.build_mild_beliefs, expected, (num_agents, 0.3, 0.7, 0.05))

    num_agents = 4
    expected = [0.25, 0.3, 0.7, 0.75]
    
    t2 = test_function(polarization.build_mild_beliefs, expected, (num_agents, 0.3, 0.7, 0.05))

    return (t1, t2)
    
def test_build_extreme_beliefs():
    num_agents = 5
    expected = [0, 0, 0, 1, 1]

    t1 = test_function(polarization.build_extreme_beliefs, expected, (num_agents,))

    num_agents = 4
    expected = [0, 0, 1, 1]
    t2 = test_function(polarization.build_extreme_beliefs, expected, (num_agents,))
    
    return (t1, t2)
    
def test_build_triple_beliefs():
    num_agents = 7
    expected = [0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0]

    t1 = test_function(polarization.build_triple_beliefs, expected, (num_agents,))
    
    num_agents = 11
    expected = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]

    t2 = test_function(polarization.build_triple_beliefs, expected, (num_agents,))

    num_agents = 9
    expected = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]

    t3 = test_function(polarization.build_triple_beliefs, expected, (num_agents,))

    return (t1, t2, t3)

def test_build_consensus_beliefs():
    num_agents = 5
    belief = 0.7
    expected = [belief] * num_agents

    t1 = test_function(polarization.build_consensus_beliefs, expected, (num_agents, belief))

    return (t1,)

def test_build_nx_beliefs():
    num_agents = 5
    expected = [0.0, 0.25, 0.5, 0.75, 1.0]

    t1 = test_function(polarization.build_nx_blf, expected, (polarization.Belief.UNIFORM, num_agents))

    num_agents = 5
    expected = [0.0, 0.16, 0.32, 0.6800000000000002, 0.8400000000000001]
    
    t2 = test_function(polarization.build_nx_blf, expected, (polarization.Belief.MILD, num_agents))

    num_agents = 4
    expected = [0.0, 0.2, 0.6000000000000001, 0.8]
    
    t3 = test_function(polarization.build_nx_blf, expected, (polarization.Belief.MILD, num_agents))

    num_agents = 5
    expected = [0.0, 0.08, 0.16, 0.8400000000000001, 0.9199999999999999]

    t4 = test_function(polarization.build_nx_blf, expected, (polarization.Belief.EXTREME, num_agents))

    num_agents = 4
    expected = [0.0, 0.1, 0.8, 0.9]
    t5 = test_function(polarization.build_nx_blf, expected, (polarization.Belief.EXTREME, num_agents))
    
    num_agents = 7
    expected = [0.0, 0.1, 0.4, 0.4666666666666667, 0.5333333333333333, 0.8, 0.9]

    t6 = test_function(polarization.build_nx_blf, expected, (polarization.Belief.TRIPLE, num_agents))
    
    num_agents = 11
    expected = [0.0, 0.05, 0.1, 0.15000000000000002, 0.4, 0.45, 0.5, 0.55, 0.8, 0.8666666666666667, 0.9333333333333333]

    t7 = test_function(polarization.build_nx_blf, expected, (polarization.Belief.TRIPLE, num_agents))

    num_agents = 9
    expected = [0.0, 0.06666666666666667, 0.13333333333333333, 0.4, 0.4666666666666667, 0.5333333333333333, 0.8, 0.8666666666666667, 0.9333333333333333]

    t8 = test_function(polarization.build_nx_blf, expected, (polarization.Belief.TRIPLE, num_agents))

    return (t1, t2, t3, t4, t5, t6, t7, t8)

def test_belief_2_distribution():
    belief_2_distribution = polarization.make_belief_2_dist_func(10)
    beliefs = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    expected = np.array([[0.05, 0.15, 0.25, 0.35, 0.45 ,0.55, 0.65, 0.75, 0.85, 0.95], [0.27272727, 0.0, 0.0, 0.0 , 0.0, 0.45454545, 0.0, 0.0, 0.0, 0.27272727]])
    
    t1 = test_function_with_numpyallclose(belief_2_distribution, expected, (beliefs,))

    return (t1,)

def test_pol_ER():
    pol_ER = polarization.make_pol_er_func(1.6, 1000)
    distribution = np.array([[0.05, 0.15, 0.25, 0.35, 0.45 ,0.55, 0.65, 0.75, 0.85, 0.95], [0.27272727, 0.0, 0.0, 0.0 , 0.0, 0.45454545, 0.0, 0.0, 0.0, 0.27272727]])
    expected = 62.29879620526804

    t1 = test_function(pol_ER, expected, (distribution,))

    return (t1,)

def test_pol_ER_discretized():
    pol_ER_discretized = polarization.make_pol_er_discretized_func(1.6, 1000, 10)
    beliefs = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
    expected = 62.298798448024755

    t1 = test_function(pol_ER_discretized, expected, (beliefs,))

    return (t1,)

def test_build_inf_graph_clique():
    num_agents = 5
    belief = 0.7
    expected = [[0.7, 0.7, 0.7, 0.7, 0.7],
       [0.7, 0.7, 0.7, 0.7, 0.7],
       [0.7, 0.7, 0.7, 0.7, 0.7],
       [0.7, 0.7, 0.7, 0.7, 0.7],
       [0.7, 0.7, 0.7, 0.7, 0.7]]

    t1 = test_function_with_numpyall(polarization.build_inf_graph_clique, expected, (num_agents, belief))

    return (t1,)

def test_build_inf_graph_2_groups_disconnected():
    num_agents = 5
    expected = [[0.3, 0.3, 0.3, 0.0, 0.0],
       [0.3, 0.3, 0.3, 0.0, 0.0],
       [0.3, 0.3, 0.3, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.3, 0.3],
       [0.0, 0.0, 0.0, 0.3, 0.3]]
    t1 = test_function_with_numpyall(polarization.build_inf_graph_2_groups_disconnected, expected, (num_agents, 0.3))
    
    num_agents = 4
    expected = [[0.3, 0.3, 0.0, 0.0],
       [0.3, 0.3, 0.0, 0.0],
       [0.0, 0.0, 0.3, 0.3],
       [0.0, 0.0, 0.3, 0.3]]
    t2 = test_function_with_numpyall(polarization.build_inf_graph_2_groups_disconnected, expected, (num_agents, 0.3))

    return (t1, t2)

def test_build_inf_graph_2_groups_faint():
    num_agents = 5
    expected = [[0.3, 0.3, 0.3, 0.1, 0.1],
       [0.3, 0.3, 0.3, 0.1, 0.1],
       [0.3, 0.3, 0.3, 0.1, 0.1],
       [0.1, 0.1, 0.1, 0.3, 0.3],
       [0.1, 0.1, 0.1, 0.3, 0.3]]

    t1 = test_function_with_numpyall(polarization.build_inf_graph_2_groups_faint, expected, (num_agents, 0.1, 0.3))
    
    num_agents = 4
    expected = [[0.3, 0.3, 0.1, 0.1],
       [0.3, 0.3, 0.1, 0.1],
       [0.1, 0.1, 0.3, 0.3],
       [0.1, 0.1, 0.3, 0.3]]

    t2 = test_function_with_numpyall(polarization.build_inf_graph_2_groups_faint, expected, (num_agents, 0.1, 0.3))

    return (t1, t2)

def test_build_inf_graph_2_influencers_balanced():
    num_agents = 5
    expected = [[0.7, 0.7, 0.7, 0.7, 0.1],
       [0.1, 0.2, 0.2, 0.2, 0.1],
       [0.1, 0.2, 0.2, 0.2, 0.1],
       [0.1, 0.2, 0.2, 0.2, 0.1],
       [0.1, 0.7, 0.7, 0.7, 0.7]]
    
    t1 = test_function_with_numpyall(polarization.build_inf_graph_2_influencers_balanced, expected, (num_agents, 0.1, 0.7, 0.2))
    
    num_agents = 4
    expected = [[0.7, 0.7, 0.7, 0.1],
       [0.1, 0.2, 0.2, 0.1],
       [0.1, 0.2, 0.2, 0.1],
       [0.1, 0.7, 0.7, 0.7]]
    
    t2 = test_function_with_numpyall(polarization.build_inf_graph_2_influencers_balanced, expected, (num_agents, 0.1, 0.7, 0.2))

    return (t1, t2)

def test_build_inf_graph_2_influencers_unbalanced():
    num_agents = 5
    expected = [[0.5, 0.5, 0.5, 0.5, 0.1],
       [0.3, 0.2, 0.2, 0.2, 0.1],
       [0.3, 0.2, 0.2, 0.2, 0.1],
       [0.3, 0.2, 0.2, 0.2, 0.1],
       [0.3, 0.4, 0.4, 0.4, 0.4]]

    t1 = test_function_with_numpyall(polarization.build_inf_graph_2_influencers_unbalanced, expected, (num_agents, 0.5, 0.4, 0.3, 0.1, 0.2))
    
    num_agents = 4
    expected = [[0.5, 0.5, 0.5, 0.1],
       [0.3, 0.2, 0.2, 0.1],
       [0.3, 0.2, 0.2, 0.1],
       [0.3, 0.4, 0.4, 0.4]]

    t2 = test_function_with_numpyall(polarization.build_inf_graph_2_influencers_unbalanced, expected, (num_agents, 0.5, 0.4, 0.3, 0.1, 0.2))

    return (t1, t2)

def test_build_inf_graph_circular():
    num_agents = 6
    i = 0.6
    expected = [[0, i, 0, 0, 0, 0],
       [0, 0, i, 0, 0, 0],
       [0, 0, 0, i, 0, 0],
       [0, 0, 0, 0, i, 0],
       [0, 0, 0, 0, 0, i],
       [i, 0, 0, 0, 0, 0]]

    t1 = test_function_with_numpyall(polarization.build_inf_graph_circular, expected, (num_agents, i))
    
    num_agents = 2
    expected = [[0, i], [i, 0]]

    t2 = test_function_with_numpyall(polarization.build_inf_graph_circular, expected, (num_agents, i))

    num_agents = 1
    expected = [[i]]

    t3 = test_function_with_numpyall(polarization.build_inf_graph_circular, expected, (num_agents, i))

    return (t1, t2, t3)

def test_function(fn, expected, params=None):
    """Tests the passed `fn` output against the `expected` result.

    Args:
      fn: The function to test.
      expected: The expected output from the function to succed the test.
      params: Optional, tuple with the params passed to `fn`.
    
    Output:
      Prints the test results.
    
    Returns:
      Boolean `True` if the equality test passes. `False` otherwise.
    """
    if params is not None:
        actual = fn(*params)
    else:
        actual = fn()
    
    return test_equality(expected, actual, fn.__name__)

def test_function_with_numpyall(fn, expected, params=None):
    """Tests the passed `fn` output against the `expected` result.

    Args:
      fn: The function to test.
      expected: The expected output from the function to succed the test.
      params: Optional, tuple with the params passed to `fn`.
    
    Output:
      Prints the test results.

    Returns:
      Boolean `True` if the equality test passes. `False` otherwise.
    """
    if params is not None:
        actual = fn(*params)
    else:
        actual = fn()
    
    result = test_equality(True, (expected == actual).all(), fn.__name__)
    if not result:
        print("Expected:", expected)
        print("  Actual:", actual)
    
    return result

def test_function_with_numpyallclose(fn, expected, params=None):
    """Test the passed `fn` output against the `expected` result.

    Args:
      fn: The function to test.
      expected: The expected output from the function to succed the test.
      params: Optional, tuple with the params passed to `fn`.
    
    Output:
      Prints the test results.
    
    Returns:
      Boolean `True` if the equality test passes. `False` otherwise.
    """
    if params is not None:
        actual = fn(*params)
    else:
        actual = fn()
    
    result = test_equality(True, np.allclose(expected, actual), fn.__name__)
    if not result:
        print("Expected:", expected)
        print("  Actual:", actual)
    
    return result

def test_update_agent_pair():
    belief_ai = 0.7
    belief_aj = 0.2
    influence = 0.2
    update_type = polarization.Update.BACKFIRE
    confbias_discount = 0.5
    backfire_belief_threshold = 0.4
    backfire_influence_threshold = 0.2
    params = [belief_ai, belief_aj, influence, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold]
    
    t1 = test_function(polarization.update_agent_pair, 0.8807970779778823, params)
    
    params[3] = polarization.Update.CLASSIC
    t2 = test_function(polarization.update_agent_pair, 0.6, params)
    
    params[3] = polarization.Update.CONFBIAS_SHARP
    t3 = test_function(polarization.update_agent_pair, 0.6499999999999999, params)
    
    params[3] = polarization.Update.CONFBIAS_SMOOTH
    t4 = test_function(polarization.update_agent_pair, 0.6499999999999999, params)

    return (t1, t2, t3, t4)

def test_update_agent_vs_all():
    agent = 3
    beliefs = [0.1, 0.2, 0.3, 0.7]
    influence = np.full((4,4),0.2)
    update_type = polarization.Update.BACKFIRE
    confbias_discount = 0.5
    backfire_belief_threshold = 0.4
    backfire_influence_threshold = 0.2
    params = [beliefs, influence, agent, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold]
    
    t1 = test_function(polarization.update_agent_vs_all, 0.77940609537099, params)
    
    params[3] = polarization.Update.CLASSIC
    t2 = test_function(polarization.update_agent_vs_all, 0.625, params)
    
    params[3] = polarization.Update.CONFBIAS_SHARP
    t3 = test_function(polarization.update_agent_vs_all, 0.6624999999999999, params)
    
    params[3] = polarization.Update.CONFBIAS_SMOOTH
    t4 = test_function(polarization.update_agent_vs_all, 0.6635, params)

    return (t1, t2, t3, t4)

def test_update_all():
    beliefs = [0.1, 0.2, 0.3, 0.7]
    influence = np.full((4,4),0.2)
    update_type = polarization.Update.BACKFIRE
    confbias_discount = 0.5
    backfire_belief_threshold = 0.4
    backfire_influence_threshold = 0.2
    params = [beliefs, influence, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold]
    expected = [0.09204064278828998, 0.1618564682943917, 0.30500000000000005, 0.77940609537099]

    t1 = test_function(polarization.update_all, expected, params)
    
    expected = [0.14500000000000002, 0.22499999999999998, 0.30500000000000005, 0.625]
    params[2] = polarization.Update.CLASSIC
    t2 = test_function(polarization.update_all, expected, params)
    
    expected = [0.13, 0.2125, 0.29500000000000004, 0.6624999999999999]
    params[2] = polarization.Update.CONFBIAS_SHARP
    t3 = test_function(polarization.update_all, expected, params)
    
    expected = [0.12450000000000001, 0.2125, 0.2995, 0.6635]
    params[2] = polarization.Update.CONFBIAS_SMOOTH
    t4 = test_function(polarization.update_all, expected, params)

    return (t1, t2, t3, t4)

if __name__ == "__main__":
    print("Running tests ...")
    print()

    tests = []

    tests.extend(test_build_uniform_beliefs())
    tests.extend(test_build_mild_beliefs())
    tests.extend(test_build_extreme_beliefs())
    tests.extend(test_build_triple_beliefs())
    tests.extend(test_build_consensus_beliefs())
    tests.extend(test_build_nx_beliefs())

    tests.extend(test_belief_2_distribution())
    tests.extend(test_pol_ER())
    tests.extend(test_pol_ER_discretized())

    tests.extend(test_build_inf_graph_clique())
    tests.extend(test_build_inf_graph_2_groups_disconnected())
    tests.extend(test_build_inf_graph_2_groups_faint())
    tests.extend(test_build_inf_graph_2_influencers_balanced())
    tests.extend(test_build_inf_graph_2_influencers_unbalanced())
    tests.extend(test_build_inf_graph_circular())

    tests.extend(test_update_agent_pair())
    tests.extend(test_update_agent_vs_all())
    tests.extend(test_update_all())

    status = "\x1b[31mFAILED\x1b[0m"
    if all(tests):
        status = "\x1b[32mok\x1b[0m"
    
    print()
    print("test result: {}. {} passed; {} failed".format(status, tests.count(True), tests.count(False)))
    print()