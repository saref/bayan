# bayanpy: Bayan Algorithm for Community Detection

bayanpy is a Python package implementing the Bayan algorithm, a community detection method capable of providing a globally optimal solution to the modularity maximization problem. Bayan can also be implemented such that it provides an approximation of the maximum modularity with a guarantee of proximity. This algorithm is theoretically grounded by the Integer Programming (IP) formulation of the modularity maximization problem and relies on an exact branch-and-cut scheme for solving the NP-complete optimization problem to global optimality.

For more information, visit the [Bayan project website](https://bayanproject.github.io/).
For a few examples on different ways you can use Bayan see this Google Colab notebook: [bayanpy Examples](https://tinyurl.com/bayancolab).


## Installation

To install bayanpy, use pip:

```
pip install bayanpy
```

## Dependencies

bayanpy has the following dependencies:

- requests>=2.25.1
- pandas>=1.3.0
- networkx>=2.6.3
- numpy>=1.21.0
- gurobipy>=9.5
- joblib>=1.1.0
- pycombo>=0.1.7

Note that pycombo requires Python version =<3.9 and therefore bayanpy has the same python version requirement.

The dependencies above will be automatically installed when you install bayanpy using pip.



### Gurobi Installation with Free Academic License

bayanpy requires Gurobi Optimizer. Gurobi is a commercial software, but it can be registered with a free academic license if the user is affiliated with an academic institution. Due to the restrictions of Gurobi, Bayan requires a (free academic) Gurobi license for processing any graph with more than sqrt(2000) ≈ 44 nodes.

Follow these steps to install Gurobi with a free academic license:

1. Download and install Python version 3.9 from [the official Python website](https://www.python.org/downloads/).
2. Register for an account on [the Gurobi registration page](https://pages.gurobi.com/registration) to get a free academic license for using Gurobi.
3. Download and install Gurobi Optimizer (version >= 9.5, latest version is recommended) from [the Gurobi downloads page](https://www.gurobi.com/downloads/gurobi-optimizer-eula/) after reading and agreeing to Gurobi's End User License Agreement.
4. Install Gurobi into Python by running the following commands (either using conda or pip) in a terminal:
- Using Conda (recommended for Anaconda users):
```
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```

- Using pip (alternative method):
```
pip install gurobipy
```

5. Request an academic license from [the Gurobi academic license page](https://www.gurobi.com/downloads/end-user-license-agreement-academic/) and install the license on your computer following the instructions given on the Gurobi license page.

For detailed installation instructions, refer to the Gurobi's [Quick Start Guides](https://support.gurobi.com/hc/en-us/articles/14799677517585-Getting-Started-with-Gurobi-Optimizer).

## Usage

You can use bayanpy as a standalone package. After installing bayanpy, you can use it directly in your Python code as follows:

```python
import networkx as nx
import bayanpy

# Create or load your networkx (undirected) graph 
graph = nx.florentine_families_graph()

# Run the Bayan algorithm
modularity, optimality_gap, community, modeling_time, solve_time = bayanpy.bayan(graph, threshold=0.001, time_allowed=60, resolution=1)
```



#### Parameters and acceptable input

- `graph`: Input graph should be an undirected networkx graph. If the graph is weighted, it must have edge attribute "weight" to store positive edge weights.
- `threshold`: The acceptable optimality gap for the algorithm to terminate. If Bayan finds a solution with a modularity value within the specified threshold of the optimal solution, it stops the search and returns the found solution. For example, setting the threshold to 0.01 means Bayan will stop when it finds a solution within 1% of the maximum modularity for that graph.
- `time_allowed`: The maximum allowed execution time in seconds for Bayan to search for a solution after the optimization model is built (formulated). Shortly after this time limit is reached, the algorithm will terminate and returns the best solution found so far, even if the optimality gap threshold is not met.
- `resolution`: The resolution parameter (gamma) used in the modularity function.

#### Returns

- `modularity`: The modularity value of the returned partition.
- `optimality_gap`: The upper bound for the percentage difference between the modularity of the returned partition and the maximum modularity.
- `community`: A nested list describing the communities (the partition).
- `modeling_time`: The seconds taken for pre-processing the input and formulating the optimization model.
- `solve_time`: The seconds taken for solving the optimization model using the Bayan algorithm.


### Example
You can find a few examples on Bayan in this Google Colab notebook: [bayanpy Examples](https://tinyurl.com/bayancolab)


## Contributing

We welcome contributions to the [Bayan project](https://bayanproject.github.io/). If you have any suggestions or encounter any issues, please feel free to open an issue or submit a pull request on this GitHub repository.

## License

bayanpy is released under the [GNU General Public License](LICENSE). For more information, please refer to the [LICENSE](LICENSE) file in the repository.




