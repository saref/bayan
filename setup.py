from setuptools import setup, find_packages

setup(
    name="bayanpy",
    version="0.1",
    packages=find_packages(),
    install_requires=["requests",
        "pandas",
        "networkx",
        "numpy",
        "gurobipy",
        "cdlib"
    ],
)
