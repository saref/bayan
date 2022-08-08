# bayan
Bayan Algorithm


# Installing Gurobi with a free academic license 

The following steps outline the process for installing Gurobi alongside a free academic license on your computer which is required to run Bayan for models with more than 2000 variables or 2000 constraints:

1-Download and install Python 3.9 (or a later version). It can be downloaded from https://www.python.org/downloads/. 

2-Register for an account on https://pages.gurobi.com/registration to get a free academic license for using Gurobi. Note that Gurobi is a commercial software, but it can be registered with a free academic license if the user is affiliated with a recognized degree-granting academic institution. This involves creating an account on Gurobi website to be able to request a free academic license in step 5.

3-Download and install Gurobi Optimizer (version 9.5 or later) which can be downloaded from https://www.gurobi.com/downloads/gurobi-optimizer-eula/ after reading and agreeing to Gurobi's End User License Agreement.

4-Install Gurobi into Python. You do this by first adding the Gurobi channel to your Anaconda channels and then installing the Gurobi package from this channel.

From a terminal, issue the following command to add the Gurobi channel to your default search list

conda config --add channels http://conda.anaconda.org/gurobi

Now issue the following command to install the Gurobi package

conda install gurobi

5-Request an academic license from https://www.gurobi.com/downloads/end-user-license-agreement-academic/ and install the license on your computer following the instructions given on Gurobi license page.

Completing these steps is explained in the following links (for version 9.5):

for windows https://www.gurobi.com/documentation/9.5/quickstart_windows/index.html

for Linux https://www.gurobi.com/documentation/9.5/quickstart_linux/index.html

for Mac OSX https://www.gurobi.com/documentation/9.5/quickstart_mac/index.html
