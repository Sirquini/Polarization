from time import perf_counter
import traceback
import unittest

import polarization
import numpy as np
import math

###################
# Basic Unit Tests
###################

def bench_test(test_fn):
    """Calls `test_fn` and reports the execution time.

    Args:
      test_fn: The test function to time.

    Output:
      Prints the test function measured time.
    """
    def timed_function(*args, **kwargs):
        counter = perf_counter()
        values = test_fn(*args, **kwargs)
        counter = perf_counter() - counter
        print("time {} ... ({})".format(test_fn.__name__, counter))
        return values
    return timed_function

class TestBuildBeliefs(unittest.TestCase):
    def test_build_uniform_beliefs(self):
        num_agents = 5
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]

        self.assertEqual(polarization.build_uniform_beliefs(num_agents), expected)

    def test_build_consensus_beliefs(self):
        num_agents = 5
        belief = 0.7
        expected = [belief] * num_agents

        self.assertEqual(polarization.build_consensus_beliefs(num_agents, belief), expected)

    def test_build_old_mild_beliefs(self):
        num_agents = 5
        expected = [0.19999999999999998, 0.25, 0.3, 0.7, 0.75]

        self.assertEqual(polarization.build_old_mild_beliefs(num_agents, 0.3, 0.7, 0.05), expected)

        num_agents = 4
        expected = [0.25, 0.3, 0.7, 0.75]

        self.assertEqual(polarization.build_old_mild_beliefs(num_agents, 0.3, 0.7, 0.05), expected)

    def test_build_old_extreme_beliefs(self):
        num_agents = 5
        expected = [0, 0, 0, 1, 1]

        self.assertEqual(polarization.build_old_extreme_beliefs(num_agents), expected)

        num_agents = 4
        expected = [0, 0, 1, 1]

        self.assertEqual(polarization.build_old_extreme_beliefs(num_agents), expected)

    def test_build_old_triple_beliefs(self):
        num_agents = 7
        expected = [0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0]

        self.assertEqual(polarization.build_old_triple_beliefs(num_agents), expected)

        num_agents = 11
        expected = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]

        self.assertEqual(polarization.build_old_triple_beliefs(num_agents), expected)

        num_agents = 9
        expected = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]

        self.assertEqual(polarization.build_old_triple_beliefs(num_agents), expected)

    def test_build_beliefs(self):
        num_agents = 5
        expected = [0.0, 0.25, 0.5, 0.75, 1.0]

        self.assertEqual(polarization.build_belief(polarization.Belief.UNIFORM, num_agents), expected)

        num_agents = 5
        expected = [0.2, 0.26666666666666666, 0.33333333333333337, 0.6, 0.7]

        self.assertEqual(polarization.build_belief(polarization.Belief.MILD, num_agents), expected)

        num_agents = 4
        expected = [0.2, 0.30000000000000004, 0.6, 0.7]

        self.assertEqual(polarization.build_belief(polarization.Belief.MILD, num_agents), expected)

        num_agents = 5
        expected = [0.0, 0.06666666666666667, 0.13333333333333333, 0.8, 0.9]

        self.assertEqual(polarization.build_belief(polarization.Belief.EXTREME, num_agents), expected)

        num_agents = 4
        expected = [0.0, 0.1, 0.8, 0.9]

        self.assertEqual(polarization.build_belief(polarization.Belief.EXTREME, num_agents), expected)

        num_agents = 7
        expected = [0.0, 0.1, 0.4, 0.4666666666666667, 0.5333333333333333, 0.8, 0.9]

        self.assertEqual(polarization.build_belief(polarization.Belief.TRIPLE, num_agents), expected)

        num_agents = 9
        expected = [0.0, 0.06666666666666667, 0.13333333333333333, 0.4, 0.4666666666666667, 0.5333333333333333, 0.8, 0.8666666666666667, 0.9333333333333333]

        self.assertEqual(polarization.build_belief(polarization.Belief.TRIPLE, num_agents), expected)

        num_agents = 11
        expected = [0.0, 0.06666666666666667, 0.13333333333333333, 0.4, 0.44, 0.48000000000000004, 0.52, 0.56, 0.8, 0.8666666666666667, 0.9333333333333333]

        self.assertEqual(polarization.build_belief(polarization.Belief.TRIPLE, num_agents), expected)

class TestPolarizationMeasure(unittest.TestCase):
    def test_belief_2_distribution(self):
        belief_2_distribution = polarization.make_belief_2_dist_func(10)
        beliefs = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
        expected = np.array([[0.05, 0.15, 0.25, 0.35, 0.45 ,0.55, 0.65, 0.75, 0.85, 0.95], [0.27272727, 0.0, 0.0, 0.0 , 0.0, 0.45454545, 0.0, 0.0, 0.0, 0.27272727]])

        np.testing.assert_allclose(belief_2_distribution(beliefs), expected)

    def test_pol_ER(self):
        pol_ER = polarization.make_pol_er_func(1.6, 1000)
        distribution = np.array([[0.05, 0.15, 0.25, 0.35, 0.45 ,0.55, 0.65, 0.75, 0.85, 0.95], [0.27272727, 0.0, 0.0, 0.0 , 0.0, 0.45454545, 0.0, 0.0, 0.0, 0.27272727]])
        expected = 62.29879620526804

        self.assertAlmostEqual(pol_ER(distribution), expected, places=10)

    def test_pol_ER_discretized(self):
        pol_ER_discretized = polarization.make_pol_er_discretized_func(1.6, 1000, 10)
        beliefs = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]
        expected = 62.298798448024755

        self.assertAlmostEqual(pol_ER_discretized(beliefs), expected, places=10)

class TestBuildInfluence(unittest.TestCase):
    def test_build_inf_graph_clique(self):
        num_agents = 5
        belief = 0.7
        expected = [[0.7, 0.7, 0.7, 0.7, 0.7],
        [0.7, 0.7, 0.7, 0.7, 0.7],
        [0.7, 0.7, 0.7, 0.7, 0.7],
        [0.7, 0.7, 0.7, 0.7, 0.7],
        [0.7, 0.7, 0.7, 0.7, 0.7]]

        np.testing.assert_equal(polarization.build_inf_graph_clique(num_agents, belief), expected)

    def test_build_inf_graph_2_groups_disconnected(self):
        num_agents = 5
        expected = [[0.3, 0.3, 0.3, 0.0, 0.0],
        [0.3, 0.3, 0.3, 0.0, 0.0],
        [0.3, 0.3, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.3, 0.3],
        [0.0, 0.0, 0.0, 0.3, 0.3]]
        np.testing.assert_equal(polarization.build_inf_graph_2_groups_disconnected(num_agents, 0.3), expected)

        num_agents = 4
        expected = [[0.3, 0.3, 0.0, 0.0],
        [0.3, 0.3, 0.0, 0.0],
        [0.0, 0.0, 0.3, 0.3],
        [0.0, 0.0, 0.3, 0.3]]
        np.testing.assert_equal(polarization.build_inf_graph_2_groups_disconnected(num_agents, 0.3), expected)

    def test_build_inf_graph_2_groups_faint(self):
        num_agents = 5
        expected = [[0.3, 0.3, 0.3, 0.1, 0.1],
        [0.3, 0.3, 0.3, 0.1, 0.1],
        [0.3, 0.3, 0.3, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.3, 0.3],
        [0.1, 0.1, 0.1, 0.3, 0.3]]

        np.testing.assert_equal(polarization.build_inf_graph_2_groups_faint(num_agents, 0.1, 0.3), expected)

        num_agents = 4
        expected = [[0.3, 0.3, 0.1, 0.1],
        [0.3, 0.3, 0.1, 0.1],
        [0.1, 0.1, 0.3, 0.3],
        [0.1, 0.1, 0.3, 0.3]]

        np.testing.assert_equal(polarization.build_inf_graph_2_groups_faint(num_agents, 0.1, 0.3), expected)

    def test_build_inf_graph_2_influencers_balanced(self):
        num_agents = 5
        expected = [[0.7, 0.7, 0.7, 0.7, 0.1],
        [0.1, 0.2, 0.2, 0.2, 0.1],
        [0.1, 0.2, 0.2, 0.2, 0.1],
        [0.1, 0.2, 0.2, 0.2, 0.1],
        [0.1, 0.7, 0.7, 0.7, 0.7]]

        np.testing.assert_equal(polarization.build_inf_graph_2_influencers_balanced(num_agents, 0.1, 0.7, 0.2), expected)

        num_agents = 4
        expected = [[0.7, 0.7, 0.7, 0.1],
        [0.1, 0.2, 0.2, 0.1],
        [0.1, 0.2, 0.2, 0.1],
        [0.1, 0.7, 0.7, 0.7]]

        np.testing.assert_equal(polarization.build_inf_graph_2_influencers_balanced(num_agents, 0.1, 0.7, 0.2), expected)

    def test_build_inf_graph_2_influencers_unbalanced(self):
        num_agents = 5
        expected = [[0.5, 0.5, 0.5, 0.5, 0.1],
        [0.3, 0.2, 0.2, 0.2, 0.1],
        [0.3, 0.2, 0.2, 0.2, 0.1],
        [0.3, 0.2, 0.2, 0.2, 0.1],
        [0.3, 0.4, 0.4, 0.4, 0.4]]

        np.testing.assert_equal(polarization.build_inf_graph_2_influencers_unbalanced(num_agents, 0.5, 0.4, 0.3, 0.1, 0.2), expected)

        num_agents = 4
        expected = [[0.5, 0.5, 0.5, 0.1],
        [0.3, 0.2, 0.2, 0.1],
        [0.3, 0.2, 0.2, 0.1],
        [0.3, 0.4, 0.4, 0.4]]

        np.testing.assert_equal(polarization.build_inf_graph_2_influencers_unbalanced(num_agents, 0.5, 0.4, 0.3, 0.1, 0.2), expected)

    def test_build_inf_graph_circular(self):
        num_agents = 6
        i = 0.6
        expected = [[1.0, i, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, i, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, i, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, i, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0, i],
        [i, 0.0, 0.0, 0.0, 0.0, 1.0]]

        np.testing.assert_equal(polarization.build_inf_graph_circular(num_agents, i), expected)

        num_agents = 2
        expected = [[1.0, i], [i, 1.0]]

        np.testing.assert_equal(polarization.build_inf_graph_circular(num_agents, i), expected)

        num_agents = 1
        expected = [[i]]

        np.testing.assert_equal(polarization.build_inf_graph_circular(num_agents, i), expected)

class TestUpdateFunctions(unittest.TestCase):
    def setUp(self):
        self.beliefs = [0.1, 0.2, 0.3, 0.7]
        self.inf_graph = np.full((4,4),0.2)
        self.expected = {
            polarization.Update.CLASSIC: [0.14500000000000002, 0.225, 0.305, 0.625],
            polarization.Update.CONFBIAS: [0.1245, 0.21250000000000002, 0.2995, 0.6635],
        }

    def test_neigbours_update(self):
        update_type = polarization.Update.CLASSIC
        self.assertEqual(list(polarization.neighbours_update(self.beliefs, self.inf_graph)), self.expected[update_type])

    def test_neigbours_cb_update(self):
        update_type = polarization.Update.CONFBIAS
        self.assertEqual(list(polarization.neighbours_cb_update(self.beliefs, self.inf_graph)), self.expected[update_type])

    def test_make_update_fn(self):
        update_type = polarization.Update.CLASSIC
        fn = polarization.make_update_fn(update_type)
        self.assertEqual(list(fn(self.beliefs, self.inf_graph)), self.expected[update_type])

        update_type = polarization.Update.CONFBIAS
        fn = polarization.make_update_fn(update_type)
        self.assertEqual(list(fn(self.beliefs, self.inf_graph)), self.expected[update_type])

class TestOldUpdateFunctions(unittest.TestCase):
    def setUp(self):
        belief_ai = 0.7
        belief_aj = 0.2
        influence = 0.2
        update_type = polarization.OldUpdate.BACKFIRE
        confbias_discount = 0.5
        backfire_belief_threshold = 0.4
        backfire_influence_threshold = 0.2

        agent = 3
        beliefs = [0.1, 0.2, 0.3, 0.7]
        inf_graph = np.full((4,4),0.2)

        self.params1 = [belief_ai, belief_aj, influence, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold]
        self.params2 = [beliefs, inf_graph, agent, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold]
        self.params3 = [beliefs, inf_graph, update_type, confbias_discount, backfire_belief_threshold, backfire_influence_threshold]

    def test_update_agent_pair(self):
        self.assertEqual(polarization.update_agent_pair(*self.params1), 0.8807970779778823)

        self.params1[3] = polarization.OldUpdate.CLASSIC
        self.assertEqual(polarization.update_agent_pair(*self.params1), 0.6)

        self.params1[3] = polarization.OldUpdate.CONFBIAS_SHARP
        self.assertEqual(polarization.update_agent_pair(*self.params1), 0.6499999999999999)

        self.params1[3] = polarization.OldUpdate.CONFBIAS_SMOOTH
        self.assertEqual(polarization.update_agent_pair(*self.params1), 0.6499999999999999)

    def test_update_agent_vs_all(self):
        self.assertEqual(polarization.update_agent_vs_all(*self.params2), 0.77940609537099)

        self.params2[3] = polarization.OldUpdate.CLASSIC
        self.assertEqual(polarization.update_agent_vs_all(*self.params2), 0.625)

        self.params2[3] = polarization.OldUpdate.CONFBIAS_SHARP
        self.assertEqual(polarization.update_agent_vs_all(*self.params2), 0.6624999999999999)

        self.params2[3] = polarization.OldUpdate.CONFBIAS_SMOOTH
        self.assertEqual(polarization.update_agent_vs_all(*self.params2), 0.6635)

    def test_update_all(self):
        expected = [0.09204064278828998, 0.1618564682943917, 0.30500000000000005, 0.77940609537099]
        self.assertEqual(polarization.update_all(*self.params3), expected)

        expected = [0.14500000000000002, 0.22499999999999998, 0.30500000000000005, 0.625]
        self.params3[2] = polarization.OldUpdate.CLASSIC
        self.assertEqual(polarization.update_all(*self.params3), expected)

        expected = [0.13, 0.2125, 0.29500000000000004, 0.6624999999999999]
        self.params3[2] = polarization.OldUpdate.CONFBIAS_SHARP
        self.assertEqual(polarization.update_all(*self.params3), expected)

        expected = [0.12450000000000001, 0.2125, 0.2995, 0.6635]
        self.params3[2] = polarization.OldUpdate.CONFBIAS_SMOOTH
        self.assertEqual(polarization.update_all(*self.params3), expected)

    def test_update_all_numpy(self):
        expected = [0.09204064278828998, 0.1618564682943917, 0.30500000000000005, 0.77940609537099]
        self.assertEqual(polarization.update_all_np(*self.params3), expected)

        expected = [0.14500000000000002, 0.22499999999999998, 0.30500000000000005, 0.625]
        self.params3[2] = polarization.OldUpdate.CLASSIC
        self.assertEqual(polarization.update_all_np(*self.params3), expected)

        expected = [0.13, 0.2125, 0.29500000000000004, 0.6624999999999999]
        self.params3[2] = polarization.OldUpdate.CONFBIAS_SHARP
        self.assertEqual(polarization.update_all_np(*self.params3), expected)

        expected = [0.12450000000000001, 0.2125, 0.2995, 0.6635]
        self.params3[2] = polarization.OldUpdate.CONFBIAS_SMOOTH
        self.assertEqual(polarization.update_all_np(*self.params3), expected)

class TestSimulationClass(unittest.TestCase):
    def test_default_constructor(self):
        blfs = polarization.build_belief(polarization.Belief.UNIFORM)
        infs = polarization.build_influence(polarization.Influence.CIRCULAR)
        simulation = polarization.Simulation(blfs, infs)
        pol_hist, _, pol = simulation.run()

        expected_pol_history = [
            24.36677041556326, 24.36677041556326, 24.36677041556326, 24.290343683819472, 23.79989503797555,
            24.083649478941968, 24.36677041556326, 24.006589242853053, 23.79989503797555, 24.006589242853053,
            23.79989503797555, 23.67880624990026, 23.79989503797555, 23.67880624990026, 23.79989503797555,
            23.67880624990026, 23.245866701217484, 23.67880624990026, 23.79989503797555, 23.67880624990026,
            23.245866701217484, 23.484678667231087, 23.79989503797555, 23.48467866723109, 23.245866701217484,
            23.484678667231087, 23.606717521496428, 23.48467866723109, 23.117928482118593, 23.484678667231087,
            23.411639872637206, 23.48467866723109, 23.117928482118593, 23.117928482118593, 23.411639872637206,
            23.245866701217484, 23.117928482118593, 23.117928482118593, 23.411639872637206, 23.245866701217484,
            22.813697682367543, 23.117928482118593, 23.411639872637206, 23.245866701217484, 22.813697682367543,
            23.117928482118593, 23.411639872637206, 23.13869447973335, 22.813697682367543, 22.967060493002414,
            23.411639872637206, 22.967060493002414, 22.813697682367543, 22.967060493002414, 23.305724295384007,
            22.967060493002414, 22.813697682367543, 22.967060493002414, 23.196643102904783, 22.967060493002414,
            22.813697682367543, 22.967060493002414, 22.98624368300415, 22.813697682367543, 22.813697682367543,
            22.813697682367543, 22.98624368300415, 22.813697682367543, 22.813697682367543, 22.813697682367543,
            22.98624368300415, 22.813697682367543, 22.67737300401307, 22.813697682367543, 22.98624368300415,
            22.813697682367543, 22.67737300401307, 22.813697682367543, 22.98624368300415, 22.813697682367543,
            22.67737300401307, 22.747405468655625, 22.813697682367543, 22.747405468655625, 22.67737300401307,
            22.747405468655625, 22.813697682367543, 22.747405468655625, 22.67737300401307, 22.747405468655625,
            22.67737300401307, 22.747405468655625, 22.67737300401307, 22.747405468655625, 22.67737300401307,
            22.636707883547253, 22.67737300401307, 22.747405468655625, 22.67737300401307, 22.523602365978626
        ]
        np.testing.assert_allclose(pol_hist, expected_pol_history)
        
        expected_pol = 22.523602365978626
        self.assertAlmostEqual(pol, expected_pol)

if __name__ == "__main__":
    unittest.main()
