from setuptools import setup, find_packages

setup(
    name="bayanpy",
    version="0.6.9",
description="Bayanpy is a powerful Python library for community detection in complex networks, designed to provide optimal or near-optimal solutions for modularity maximization. It features a highly efficient branch-and-cut algorithm and is backed by Integer Programming (IP) formulations.",
    packages=find_packages(),
    install_requires=["requests",
        "pandas",
        "networkx",
        "numpy",
        "gurobipy",
	 "joblib",
        "pycombo"
    ],
)
