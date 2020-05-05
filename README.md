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
import polarization

# By default the number of agents is 100,
# but can be set as an argument.
beliefs_vec = polarization.build_belief(polarization.Belief.UNIFORM, num_agents=100)

# For a CLIQUE influence we could change the influence value,
# defaults to 0.5, by passing the general_belief argument.
inf_graph = polarization.build_influence(polarization.Inflluence.CLIQUE, general_belief=0.5)

# By default max_time is set to 100, and the update function to
# Update.CLASSIC
polarization_history, belief_history, polarization = run_till_convergence(belief_vec, inf_graph, update_type=polarization.Update.CLASSIC)
```

We may want to plot the evolution of polarization over time:
```python
import matplotlib.pyplot as plt

plt.plot(polarization_history)
plt.show()
```

For more examples you can look at code in the Jupyter Notebooks of this repository. `New Simulations.ipynb` for the usage of the `polarization` module, and `Simulations of Polarization.ipynb` for the theory behind.