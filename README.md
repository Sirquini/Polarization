# A model for the dynamics of polarization

All the functions and structures needed are located in `polarization.py`. Implemented for Python 3.

## Dependencies

This project depends on [NumPy](https://numpy.org/index.html) which may be installed from a console using pip:

```
pip install numpy
```

Optionally, we recommend installing [Matplotlib](https://matplotlib.org/) for creating graphs and charts, , and [Jupyter Notebooks](https://jupyter.org/index.html) for running simulations interactively.

```
pip install matplotlib
```

```
pip install notebook
```

## Basic Usage

You only need to import the `polarization` module, which provides:

- `run_till_convergence`: Runs a simulation until convergence is achieved, or `max_time` is reached (defaults to `100`), given an initial belief vector, influence graph, and update function (defaults to `CLASSIC`).

- `build_belief`: Helper function that returns a belief vector based on a `Belief` scenario type (`UNIFORM`, `MILD`, `EXTREME`, `TRIPLE`, `CONSENSUS`).

- `build_influence`: Helper function that returns an influence graph based on an `Influence` scenario type (`CLIQUE`, `GROUP_2_DISCONECTED`, `GROUP_2_FAINT`, `INFLUENCERS_2_BALANCED`, `INFLUENCERS_2_UNBALANCED`).

As such, running a simulation with mostly default parameters, is as simple as:

```python
import polarization as plr

# By default the number of agents is 100,
# but can be set as an argument.
beliefs_vec = plr.build_belief(plr.Belief.UNIFORM, num_agents=100)

# For a CLIQUE influence we could change the influence value,
# defaults to 0.5, by passing the general_belief argument.
inf_graph = plr.build_influence(plr.Inflluence.CLIQUE, general_belief=0.5)

# By default max_time is set to 100, and the update function to
# Update.CLASSIC
polarization_history, belief_history, pol = run_till_convergence(belief_vec, inf_graph, update_type=plr.Update.CLASSIC)
```

We may want to plot the evolution of plr over time:
```python
import matplotlib.pyplot as plt

plt.plot(polarization_history)
plt.show()
```

For more examples you can look at code in the Jupyter Notebooks of this repository. `New Simulations.ipynb` for the usage of the `polarization` module, and `Simulations of Polarization.ipynb` for the theory behind.

## Alternative `Simulation` Class

You can also run simulations creating an instance of the `Simulation` class and either its `run(max_time=100, stop_at_convergence=True)` method or iterating it like any other python iterator. As always you need to provide the initial belief state, an influence graph, and the belief update function. Optionally you can define an alternative function for computing polarization, other than Esteban-Ray.

By default, Simulation uses the `CLASSIC` update function, but can be changed by passing an `update_fn` parameter. Also by default, the polarization measure is Esteban-Ray with 201 bins to discretize the belief state into a distribution, you can change the polarization measure by passing a `pol_measure` function, or just change the `num_bins`.

```python
import polarization as plr

simulation = plr.Simulation(
    plr.build_belief(plr.Belief.UNIFORM),
    plr.build_influence(plr.Influence.CLIQUE),
    plr.make_update_fn(plr.Update.CLASSIC)
)

polarization_history, belief_history, pol = simulation.run()
```

## Helper Functions

For creating the initial beliefs:

- `build_inf_graph_clique(num_agents, belief_value)`
- `build_inf_graph_2_groups_disconnected(num_agents, belief_value)`
- `build_inf_graph_2_groups_faint(num_agents, weak_belief_value, strong_belief_value)`
- `build_inf_graph_2_influencers_balanced(num_agents, influencers_incoming_value, influencers_outgoing_value, others_belief_value)`
- `build_inf_graph_2_influencers_unbalanced(num_agents, influencers_outgoing_value_first, influencers_outgoing_value_second, influencers_incoming_value_first, influencers_incoming_value_second, others_belief_value)`

> It is recommended to just use the general `build_belief(belief_type, **kwargs)`.

For creating the influence graph:

- `build_inf_graph_clique(num_agents, belief_value)`
- `build_inf_graph_2_groups_disconnected(num_agents, belief_value)`
- `build_inf_graph_2_groups_faint(num_agents, weak_belief_value, strong_belief_value)`
- `build_inf_graph_2_influencers_balanced(num_agents, influencers_incoming_value, influencers_outgoing_value, others_belief_value)`
- `build_inf_graph_2_influencers_unbalanced(num_agents, influencers_outgoing_value_first, influencers_outgoing_value_second, influencers_incoming_value_first, influencers_incoming_value_second, others_belief_value)`

> It is recommended to just use the general `build_influence(inf_type, **kwargs)`.

For creating an update function to use with the `Simulation` class:

- `make_update_fn(update_type, **kwarg)`: The resulting function is of type `update_fn(belief_vec, inf_graph)`.

For customizing the polarization measure with the `Simulation` class:

- `make_pol_er_discretized_func(alpha, K, num_bins)`: The resulting function is of type `pol_ER_discretized(belief_state)`.