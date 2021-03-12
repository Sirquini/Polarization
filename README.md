# A Model for the Dynamics of Polarization

[![CI](https://github.com/Sirquini/Polarization/actions/workflows/ci.yml/badge.svg)](https://github.com/Sirquini/Polarization/actions/workflows/ci.yml)

All the functions and structures needed are located in `polarization.py`. Implemented for Python 3.

## Dependencies

This project depends on [NumPy](https://numpy.org/index.html) (tested on 1.19.0) which may be installed from a console using pip:

```
python3 -m pip install numpy
```

Optionally, we recommend installing [Matplotlib](https://matplotlib.org/) for creating graphs and charts, and [Jupyter Notebooks](https://jupyter.org/index.html) for running simulations interactively.

```
python3 -m pip install matplotlib notebook
```

To install all dependencies you can also run:

```
python3 -m pip install -r requirements.txt
```

## Running Simulations

If you want to try, or recreate, the simulations in `Simulations.ipynb` for yourself, download this repository, install the corresponding dependencies, and run the jupyter notebook server with access to `polarization.py`. For example, in the same folder as the `polarization.py` file is located, run:

```
jupyter notebook
```

Select `Simulations.ipynb`, and make sure to set it as trusted, or just re-run the desired cells.

## The `Simulation` Class

You can run simulations creating an instance of the `Simulation` class and either its `run(max_time=100, smart_stop=True)` method or iterating it like any other python iterator. As always, you need to provide the initial belief state, an influence graph, and a belief update function. Optionally you can define an alternative function for computing polarization, other than Esteban-Ray.

By default, `Simulation` uses the `CLASSIC` (or regular) update function, but can be changed by passing an `update_fn` parameter. Also by default, the polarization measure is Esteban-Ray with 5 bins to discretize the belief state into a distribution, you can change the polarization measure by passing a `pol_measure` function, or just change the `num_bins` argument.

```python
import polarization as plr

simulation = plr.Simulation(
    plr.build_belief(plr.Belief.UNIFORM),
    plr.build_influence(plr.Influence.CLIQUE),
    plr.make_update_fn(plr.Update.CLASSIC)
)

polarization_history, belief_history, pol = simulation.run()
```

We may want to plot the evolution of polarization over time:
```python
import matplotlib.pyplot as plt

plt.plot(polarization_history)
plt.show()
```

### Polarization Measure and Update Function for `Simulation`

For creating an update function to use with the `Simulation` class:

- `make_update_fn(update_type)`: The resulting function is of type `update_fn(belief_vec, inf_graph)`.

For customizing the polarization measure with the `Simulation` class:

- `make_pol_er_discretized_func(alpha, K, num_bins)`: The resulting function is of type `pol_ER_discretized(belief_state)`.

For more examples you can look at code in the Jupyter Notebooks of this repository, e.g. `Simulations.ipynb` for the usage of the `polarization` module.

## Initial Belief Configurations

The definition of `Belief`.{`UNIFORM`, `MILD`, `EXTREME`, `TRIPLE`} is as follows:

There is a new function that allows us to generate new initial belief configurations based on a 5 bins Esteban-Ray Polarization measure, evently distributing all agents in clusters between the [0, 1] interval.

| Belief      | [0, 0.2) | [0.2, 0.4) | [0.4, 0.6) | [0.6, 0.8) | [0.8, 1] |
| ----------- | :------: | :--------: | :--------: | :--------: | :------: |
| UNIFORM     | o | o | o | o | o |
| MILD        |   | o |   | o |   |
| EXTREME     | o |   |   |   | o |
| TRIPLE      | o |   | o |   | o |

To generate such configurations `build_belief` is provided.

## Alternative Functions for Simulations

Apart from the `Simulation` class, the `polarization` module provides:

- `build_belief`: Function that returns a belief vector based on a `Belief` scenario type (`UNIFORM`, `MILD`, `EXTREME`, `TRIPLE`, `CONSENSUS`), this function produces the new definition of a Belief configuration, as such, it is the recommended way of building initial belief vectors.

- `build_influence`: Helper function that returns an influence graph based on an `Influence` scenario type (`CLIQUE`, `GROUP_2_DISCONECTED`, `GROUP_2_FAINT`, `INFLUENCERS_2_BALANCED`, `INFLUENCERS_2_UNBALANCED`, `CIRCULAR`).

- `run_until_stable`: **Legacy** helper function that runs a simulation until convergence is achieved, or `max_time`
 is reached (defaults to `100`), given an initial belief vector, influence graph,
 and the old definition of an update function (defaults to `CLASSIC`, see the `OldUpdate` enum). **Deprecated.** Produces different output than calling the `Simulation`'s `run()` method.

- `build_old_belief`: **Legacy** helper function that returns a belief vector based on a `Belief` scenario type (`UNIFORM`, `MILD`, `EXTREME`, `TRIPLE`, `CONSENSUS`). **Deprecated.** Produces different output than using `build_belief`.

## Helper Functions

For creating the initial beliefs:

- `build_belief(belief_type: Belief, num_agents, **kwargs)`

For creating the influence graph:

- `build_influence(inf_type: Influence, num_agents, **kwargs)`

Alternatively:

- `build_inf_graph_clique(num_agents, belief_value)`
- `build_inf_graph_2_groups_disconnected(num_agents, belief_value)`
- `build_inf_graph_2_groups_faint(num_agents, weak_belief_value, strong_belief_value)`
- `build_inf_graph_2_influencers_balanced(num_agents, influencers_incoming_value, influencers_outgoing_value, others_belief_value)`
- `build_inf_graph_2_influencers_unbalanced(num_agents, influencers_outgoing_value_first, influencers_outgoing_value_second, influencers_incoming_value_first, influencers_incoming_value_second, others_belief_value)`
- `build_inf_graph_circular(num_agents, value)`

> It is recommended to use the general `build_influence(*args, **kwargs)`.

For creating the legacy initial beliefs:

- `build_uniform_beliefs(num_agents)`
- `build_old_mild_beliefs(num_agents, low_pole, high_pole, step)`
- `build_old_extreme_beliefs(num_agents)`
- `build_old_triple_beliefs(num_agents)`
- `build_consensus_beliefs(num_agents, belief)`

> It is recommended to use the general `build_belief(belief_type: Belief, num_agents, **kwargs)` for the current configurations.

> It is recommended to use the general `build_old_belief(belief_type: Belief, num_agents, **kwargs)` for the old configurations.
