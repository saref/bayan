#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Install Gurobi

# get_ipython().run_line_magic('pip', 'install -i https://pypi.gurobi.com gurobipy')
    
# Install cdlib

# get_ipython().run_line_magic('pip', 'install cdlib')


# In[ ]:


# importing necessary modules
import requests, zipfile, shutil, json
import pandas as pd
import networkx as nx
from io import BytesIO
import numpy as np
import time
import multiprocessing
from gurobipy import *
from cdlib import algorithms
from networkx.algorithms.connectivity import minimum_st_node_cut


# In[ ]:


#Load Graph from network name
def get_graph_from_network_name(network_name, sub_name=None):
    """
    Method to create a networkx Graph from network names listed at https://networks.skewed.de/
    """
    # Defining the network information url
    info_url = "https://networks.skewed.de/api/net/%s" % network_name
    req = requests.get(info_url)
    info_dict = json.loads(req.text)
    if sub_name is None:
        net = info_dict['nets'][0]
    else:
        net = sub_name

    # Check and store the names of the available node/edge properties and whether the graph is directed
    if net == network_name:
        vertex_properties = []
        for prop in info_dict['analyses']["vertex_properties"]:
            vertex_properties.append(prop[0])

        edge_properties = []
        for prop in info_dict['analyses']["edge_properties"]:
            edge_properties.append(prop[0])
        
        is_directed = info_dict['analyses']['is_directed']
    else:
        vertex_properties = []
        for prop in info_dict['analyses'][net]["vertex_properties"]:
            vertex_properties.append(prop[0])

        edge_properties = []
        for prop in info_dict['analyses'][net]["edge_properties"]:
            edge_properties.append(prop[0])
        
        is_directed = info_dict['analyses'][net]['is_directed']
    
    # Defining the zip file URL
    url = "https://networks.skewed.de/net/%s/files/%s.csv.zip" % (network_name, net)

    # Downloading the file by sending the request to the URL
    req = requests.get(url)

    # extracting the zip file contents
    z= zipfile.ZipFile(BytesIO(req.content))
    if os.path.isdir(network_name + "_files"):
        shutil.rmtree(network_name + "_files")
    # creating a dir to extract the files into
    os.mkdir(network_name + "_files")
    z.extractall(os.getcwd() + '/%s_files'%network_name)
    
    # read in the dataframes of the nodes and edges of the network
    nodes_df = pd.read_csv(network_name + "_files/nodes.csv")
    edges_df = pd.read_csv(network_name + "_files/edges.csv")
    
    # Create the graph
    G = nx.Graph()
    
    # Add the nodes and their attributes to the graph
    node_attrs = {}
    for row in range(nodes_df.shape[0]):
        G.add_node(nodes_df.iloc[row][0])
        vertex_dict = {}
        for i in range(len(vertex_properties)):
            if vertex_properties[i] == '_pos':
                vertex_dict[vertex_properties[i]] = [float(y) for y in nodes_df.iloc[row][i+1][7:-2].split(",")]
            else:
                vertex_dict[vertex_properties[i]] = nodes_df.iloc[row][i+1]
        node_attrs[nodes_df.iloc[row][0]] = vertex_dict
    nx.set_node_attributes(G, node_attrs)
    
    # Add the edges and their attributes to the graph
    edge_attrs = {}
    is_multigraph = False
    for row in range(edges_df.shape[0]):
        is_multigraph = is_multigraph or G.has_edge(edges_df.iloc[row][0], edges_df.iloc[row][1])
        G.add_edge(edges_df.iloc[row][0], edges_df.iloc[row][1])
        edge_dict = {}
        for i in range(len(edge_properties)):
            edge_dict[edge_properties[i]] = edges_df.iloc[row][i+2]
        edge_attrs[(edges_df.iloc[row][0], edges_df.iloc[row][1])] = edge_dict
    nx.set_edge_attributes(G, edge_attrs)
    
    if is_multigraph:
        print("%s is a multigraph but has been loaded as a simple graph" % network_name)
    
    shutil.rmtree(network_name + "_files")
    return G


# In[ ]:


def get_local_clustering_coefficient(G, node: int):
    """
    Returns the clustering coefficient for the input node in the input graph
    """
    neighbours = nx.adjacency_matrix(G)[node].indices
    if neighbours.shape[0] <= 1:
        return 0.0
    num_possible_edges = ((neighbours.shape[0])*(neighbours.shape[0]-1)) / 2
    num_actual_edges = 0
    for neighbour in neighbours:
        num_actual_edges += np.intersect1d(neighbours, nx.adjacency_matrix(G)[neighbour].indices).shape[0]
    num_actual_edges = num_actual_edges / 2
    return num_actual_edges / num_possible_edges

def clique_filtering(G, resolution):
    """
    Returns G' which is a clique reduction on the input graph G 
    """
    lcc_dict = {}
    for node in G.nodes():
        lcc_dict[node] = get_local_clustering_coefficient(G, node)
    shrink_dict = {}
    for node in G.nodes():
        skip = False
        for n in G.nodes():
            if n in shrink_dict and shrink_dict[n] == node and n != node:
                skip = True
        if skip:
            shrink_dict[node] = node
            continue
        neighbours = nx.adjacency_matrix(G)[node].indices
        if neighbours.shape[0] == 1:
            shrink_dict[node] = neighbours[0]
            continue
        count_of_ones = 0
        count_of_not_one = 0
        not_one_neighbour = -1
        for neighbour in neighbours:
            if lcc_dict[neighbour] == 1:
                count_of_ones += 1
            else:
                count_of_not_one += 1
                not_one_neighbour = neighbour
        if count_of_ones == neighbours.shape[0] - 1 and count_of_not_one == 1:
            shrink_dict[node] = not_one_neighbour
        if node not in shrink_dict.keys():
            shrink_dict[node] = node
    G_prime = G.copy()
    for key in shrink_dict:
        if key != shrink_dict[key]:
            G_prime.nodes[shrink_dict[shrink_dict[key]]]['super node of'].extend(G_prime.nodes[key]['super node of'])
            edges_to_del = G_prime.edges(key)
            total_weight = 0
            for edge in edges_to_del:
                if 'actual_weight' in G_prime.edges[edge]:
                    total_weight += 2*G_prime.edges[edge]['actual_weight']
                else:
                    total_weight += 2
            if G_prime.has_edge(shrink_dict[shrink_dict[key]], shrink_dict[shrink_dict[key]]):
                G_prime.edges[(shrink_dict[shrink_dict[key]], shrink_dict[shrink_dict[key]])]['actual_weight'] += total_weight
                G_prime.edges[(shrink_dict[shrink_dict[key]], shrink_dict[shrink_dict[key]])]['constrained_modularity'] = False
            else:
                G_prime.add_edge(shrink_dict[shrink_dict[key]], shrink_dict[shrink_dict[key]], actual_weight=total_weight, weight=0, constrained_modularity=False)
            G_prime.remove_node(key)
    G_prime = nx.convert_node_labels_to_integers(G_prime)
    ModularityMatrix = get_modularity_matrix(G_prime, resolution)
    for edge in G_prime.edges():
        G_prime.edges[edge[0], edge[1]]["weight"] = ModularityMatrix[edge[0], edge[1]]
    return G_prime


# In[ ]:


def find_in_list_of_list(mylist, char):
    for sub_list in mylist:
        if char in sub_list:
            return (mylist.index(sub_list))
    raise ValueError("'{char}' is not in list".format(char = char))


# In[ ]:


def model_to_communities(var_vals, Graph):
    """
    Method that outputs communities with input model variable values and the Graph
    """
    clustered = []
    
    for v in var_vals:
        if var_vals[v]!=1:
            clustered.append((v).split(","))
    i=0
    visited_endpoints=[]
    group=[]
    for pair in clustered:
        if pair[0] not in visited_endpoints and pair[1] not in visited_endpoints and pair[0] != pair[1]:
            visited_endpoints.append(pair[0])
            visited_endpoints.append(pair[1])
            group.append([])
            (group[i]).append(pair[0])
            (group[i]).append(pair[1])
            i=i+1
        if pair[0] not in visited_endpoints and pair[1] in visited_endpoints:
            index_one=find_in_list_of_list(group, pair[1])
            (group[index_one]).append(pair[0])
            visited_endpoints.append(pair[0])
        if pair[1] not in visited_endpoints and pair[0] in visited_endpoints:
            index_zero=find_in_list_of_list(group, pair[0])
            (group[index_zero]).append(pair[1])
            visited_endpoints.append(pair[1])
        if pair[0] in visited_endpoints and pair[1] in visited_endpoints:
            index_zero=find_in_list_of_list(group, pair[0])
            index_one=find_in_list_of_list(group, pair[1])
            if index_zero!=index_one:
                group[index_zero]=group[index_zero]+group[index_one]
                del group[index_one]
                i=i-1

    for node in (Graph).nodes():
        if str(node) not in visited_endpoints:
            group.append([str(node)])

    for i in range(len(group)):
        for j in range(len(group[i])):
            group[i][j] = int(group[i][j])

    for c in range(len(group)):
        group[c].sort()
    group.sort()
    
    return group


# In[ ]:


def decluster_communities(group, Graph, isolated_nodes):
    """
    Method to get communities based on the original graph. Note, input Graph is the reduced graph.
    """
    group_declustered = []
    for comm in group:
        new_comm = []
        for node in comm:
            if 'super node of' in Graph.nodes[int(node)]:
                node_list = Graph.nodes[int(node)]['super node of']
                new_comm = new_comm + node_list
            else:
                new_comm.append(int(node))
        group_declustered.append(new_comm)
    for n in isolated_nodes:
        group_declustered.append([n])
    for c in range(len(group_declustered)):
        group_declustered[c].sort()
    group_declustered.sort()
    return group_declustered


# In[ ]:


def lp_formulation (Graph, AdjacencyMatrix, ModularityMatrix, size, order, isolated_nodes, warmstart=int(0), branching_priotiy=int(0)):
    """
    Method to create the LP model and run it for the root node
    """

    objectivevalue=0
    solveTime=0
    effectiveBranchingFactors=0
    number_of_subsets=[]
    communities=[]

    formulation_time_start = time.time()
    list_of_cut_triads=[]
    for i in (Graph).nodes():
        for j in range(i+1, len((Graph).nodes())):
            removed_edge = False
            if Graph.has_edge(i, j):
                removed_edge = True
                attr_dict = Graph.edges[i, j]
                Graph.remove_edge(i, j)
            minimum_vertex_cut=minimum_st_node_cut(Graph, i, j)
            for k in minimum_vertex_cut:
                list_of_cut_triads.append(list(np.sort([i,j,k])))
            if removed_edge:
                Graph.add_edge(i, j, weight=attr_dict["weight"], constrained_modularity=attr_dict["constrained_modularity"], actual_weight=attr_dict["actual_weight"])

    x={}

    model = Model("Modularity maximization")
    model.setParam(GRB.param.OutputFlag, 0)
    model.setParam(GRB.param.Method, -1)
    model.setParam(GRB.Param.Crossover, 0) 
    model.setParam(GRB.Param.Threads, min(64,multiprocessing.cpu_count()))
#     model.setParam("OutputFlag", 0)

    print('Adding variables...')
    for i in range(len(Graph.nodes())):
        for j in range(i+1, len(Graph.nodes())):
            x[(i,j)] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=str(i)+','+str(j))

    model.update()
    
    OFV = 0 
    for i in range(len(Graph.nodes())):
        for j in range(i+1, len(Graph.nodes())):
            OFV += ModularityMatrix[i, j] * (1 - x[(i, j)])

    model.setObjective(OFV, GRB.MAXIMIZE)

    print('Adding constraints...')
    for [i,j,k] in list_of_cut_triads:
        model.addConstr(x[(i,k)] <= x[(i, j)] + x[(j, k)], 'triangle1'+','+str(i)+','+str(j)+','+str(k))
        model.addConstr(x[(i,j)] <= x[(i, k)] + x[(j, k)], 'triangle2'+','+str(i)+','+str(j)+','+str(k))
        model.addConstr(x[(j,k)] <= x[(i, j)] + x[(i, k)], 'triangle3'+','+str(i)+','+str(j)+','+str(k))
    formulation_time = time.time() - formulation_time_start

    model.update()

    # Using a warm start
    # A known partition can be used as a starting point to "warm-start" the algorithm.
    # It can be provided to the model here by giving start values to the decision variables:
    # If this optional step is skipped, you should comment out these five lines
    if warmstart==1:
        #Solution from Combo algorithm to be used as warm-start
        partition = algorithms.pycombo(Graph[index])
        community_combo=list(partition.communities)
        for i in (Graph).nodes():
            for j in filter(lambda x: x>i, (Graph).nodes()):
                if find_in_list_of_list(community_combo,i)==find_in_list_of_list(community_combo,j):
                    x[(i,j)].start = 0
                else:
                    x[(i,j)].start = 1   

    #branching priority is based on total degrees of pairs of nodes
    if branching_priotiy == 1:
        neighbors={}
        Degree=[]
        for i in range(len(Graph.nodes())):
            for j in range(i+1, len(Graph.nodes())):
                neighbors[i] = list((Graph)[i])
                neighbors[j] = list((Graph)[j])
                Degree.append(len(neighbors[i])+len(neighbors[j]))
        model.setAttr('BranchPriority',model.getVars()[:],Degree)
        model.update()

    start_time = time.time()
    model.optimize()
    solveTime = (time.time() - start_time) 

    obj = model.getObjective()

    objectivevalue = np.round(((2*obj.getValue()+(ModularityMatrix.trace()[0, 0]))/np.sum(AdjacencyMatrix)), 8)


#     print('Instance 0',': modularity equals',objectivevalue) 

    if (model.NodeCount)**(1/((size)+2*(order))) >= 1:
        effectiveBranchingFactors = ((model.NodeCount)**(1/((size)+2*(order))))  
    
    var_vals = {}
    for var in model.getVars():
        var_vals[var.varName] = var.x
        
    communities = model_to_communities(var_vals, Graph)

#     print("Communities: ", communities)

    communities_declustered = decluster_communities(communities, Graph, isolated_nodes)
#     print("Communities de-clustered: ", communities_declustered)

    

#     print("\n")
#     print("Q*:")
#     print(objectivevalue)

#     print("solve time:",solveTime)
    
    return objectivevalue, var_vals, model, list_of_cut_triads, formulation_time, solveTime


# In[ ]:


def run_lp(model, Graph, fixed_ones, fixed_zeros, resolution):
    """
    Run the LP based on model and the original Graph as input
    """
    ModularityMatrix = get_modularity_matrix(Graph, resolution)
    AdjacencyMatrix = nx.to_numpy_matrix(Graph, weight='actual_weight')
    
    for var_name in fixed_ones:
        var = model.getVarByName(var_name)
        var.setAttr("LB", 1.0)
    model.update()
    for var_name in fixed_zeros:
        var = model.getVarByName(var_name)
        var.setAttr("UB", 0.0)
    model.update()
    
    start_time = time.time()
    model.optimize()
    solveTime = (time.time() - start_time) 

    obj = model.getObjective()
    try:
        obj_val = obj.getValue()
    except AttributeError as error:
        return -1, -1, model

    objectivevalue = np.round((((2*obj_val)+(ModularityMatrix.trace()[0, 0]))/np.sum(AdjacencyMatrix)), 8)


    var_vals = {}
    for var in model.getVars():
        var_vals[var.varName] = var.x

#     print("\n")
#     print("Q*:")
#     print(objectivevalue)

#     print("solve time:",solveTime)
    
    return objectivevalue, var_vals, model


# In[ ]:


def reset_model_varaibles(model, fixed_ones, fixed_zeros):
    for var_name in fixed_ones:
        var = model.getVarByName(var_name)
        var.setAttr("LB", 0.0)
    model.update()
    for var_name in fixed_zeros:
        var = model.getVarByName(var_name)
        var.setAttr("UB", 1.0)
    model.update()
    return model


# In[ ]:


def calculate_modularity(community, Graph, resolution):
    """
    Method that calculates modularity for input community partition on input Graph
    """
    ModularityMatrix = get_modularity_matrix(Graph, resolution)
    AdjacencyMatrix = nx.to_numpy_matrix(Graph, weight="actual_weight")
    OFV=0
    for item in community:
        if len(item)>1:
            for i in range(0,len(item)):
                for j in range(i+1,len(item)):
                    OFV = OFV + 2*(ModularityMatrix[item[i],item[j]])
    OFV = OFV + ModularityMatrix.trace()[0, 0]
    return np.round(OFV/(np.sum(AdjacencyMatrix)), 8)

def find_violating_triples(Graph, var_vals, list_of_cut_triads):
    """
    Returns a dictionary whose key is a violated constrained and value is the sum
    """
    violated_triples_sums = {}
    for [i, j, k] in list_of_cut_triads:
        triple_sum = var_vals[str(i) + "," + str(j)] + var_vals[str(j) + "," + str(k)] + var_vals[str(i) + "," + str(k)]
        if 0 < triple_sum < 2:
            violated_triples_sums[(i, j, k)] = triple_sum
    return violated_triples_sums

def get_best_triple(violated_triples_sums, node, orig_g):
    """
    Returns the constraint with the most in common with the previous nodes constraint
    """    
    num_nodes = len(list(orig_g.nodes()))
    score_list = []
    violated_triples = list(violated_triples_sums.keys())
    for triple in violated_triples:
        new_triple = [-1, -1, -1]
        value_zero = orig_g.nodes[triple[0]]['super node of'][0]
        value_one = orig_g.nodes[triple[1]]['super node of'][0]
        value_two = orig_g.nodes[triple[2]]['super node of'][0]
        for n in node.graph.nodes():
            if value_zero in node.graph.nodes[n]["super node of"]:
                new_triple[0] = n
            if value_one in node.graph.nodes[n]["super node of"]:
                new_triple[1] = n
            if value_two in node.graph.nodes[n]["super node of"]:
                new_triple[2] = n
            if new_triple[0] != -1 and new_triple[1] != -1 and new_triple[2] != -1:
                break
        total_score = 0
        for t in range(3):
            alpha = 0
            for i in range(triple[t]+1, num_nodes):
                variable = str(triple[t])+","+str(i)
                if variable in node.get_fixed_ones():
                    alpha += 1
                if variable in node.get_fixed_zeros():
                    alpha += 1
            beta = 0
            if node.parent is not None:
                for constr in node.constraints:
                    if triple[t] in constr[:3]:
                        beta += 1
            delta = node.graph.degree(new_triple[t], weight="actual_weight")
            score = 1 - np.exp(-alpha) + beta + (delta/len(list(node.graph.nodes())))
            total_score += score
        score_list.append(total_score)
    sum_of_scores = np.sum(score_list)
    probability_list = [x/sum_of_scores for x in score_list]
#     np.random.seed(5323)
    index = np.random.choice(len(violated_triples), p=probability_list)
    return violated_triples[index][:3]
            
    
def is_integer_solution(Graph, var_vals):
    """
    Return whether all the varaible values are integer
    """
    for i in range(len(Graph.nodes())):
        for j in range(i+1, len(Graph.nodes())):
            if var_vals[str(i) + "," + str(j)] != 1 and var_vals[str(i) + "," + str(j)] != 0:
                return False
    return True

def reduce_triple(G, triple, orig_g, resolution):
    """
    Reduces G by creating a supernode for nodes in triple (for left branching). 
    Returns the reduced graph and the additional edges added for running combo
    """
    new_triple = [-1,-1,-1]
    value_zero = orig_g.nodes[triple[0]]['super node of'][0]
    value_one = orig_g.nodes[triple[1]]['super node of'][0]
    value_two = orig_g.nodes[triple[2]]['super node of'][0]
    for node in G.nodes():
        if value_zero in G.nodes[node]["super node of"]:
            new_triple[0] = node
        if value_one in G.nodes[node]["super node of"]:
            new_triple[1] = node
        if value_two in G.nodes[node]["super node of"]:
            new_triple[2] = node
    triple = new_triple
    Graph = G.copy()
    self_weight = 0
    for u in range(3):
        for v in range(u, 3):
            if Graph.has_edge(triple[u], triple[v]):
                if 'actual_weight' in Graph.edges[triple[u], triple[v]]:
                    self_weight += Graph.edges[triple[u], triple[v]]["actual_weight"]
                else:
                    self_weight += 1
    if Graph.has_edge(triple[0], triple[0]):
        Graph.edges[triple[0], triple[0]]["actual_weight"] = self_weight
        curr = Graph.nodes[triple[0]]['super node of']
        new1 = Graph.nodes[triple[1]]['super node of']
        new2 = Graph.nodes[triple[2]]['super node of']
        super_list = list(set(curr + new1 + new2))
        super_list.sort()
        Graph.nodes[triple[0]]['super node of'] = super_list
    else:
        Graph.add_edge(triple[0], triple[0])
        Graph.edges[triple[0], triple[0]]["actual_weight"] = self_weight
        Graph.edges[triple[0], triple[0]]['constrained_modularity'] = False
        curr = Graph.nodes[triple[0]]['super node of']
        new1 = Graph.nodes[triple[1]]['super node of']
        new2 = Graph.nodes[triple[2]]['super node of']
        super_list = list(set(curr + new1 + new2))
        super_list.sort()
        Graph.nodes[triple[0]]['super node of'] = super_list
    
    if triple[0] != triple[1]:
        edge_list = list(Graph.edges(triple[1]))
        for edge in edge_list:
            if edge[1] not in [triple[0], triple[2]]:
                if Graph.has_edge(triple[0], edge[1]):
                    Graph.edges[triple[0], edge[1]]['actual_weight'] += Graph.edges[edge]['actual_weight']
                else:
                    Graph.add_edge(triple[0], edge[1])
                    Graph.edges[triple[0], edge[1]]['actual_weight'] = Graph.edges[edge]['actual_weight']
                    Graph.edges[triple[0], edge[1]]['constrained_modularity'] = False
        Graph.remove_node(triple[1])
    
    if triple[0] != triple[2] and triple[1] != triple[2]:
        edge_list = list(Graph.edges(triple[2]))
        for edge in edge_list:
            if edge[1] not in [triple[0], triple[1]]:
                if Graph.has_edge(triple[0], edge[1]):
                    Graph.edges[triple[0], edge[1]]['actual_weight'] += Graph.edges[edge]['actual_weight']
                else:
                    Graph.add_edge(triple[0], edge[1])
                    Graph.edges[triple[0], edge[1]]['actual_weight'] = Graph.edges[edge]['actual_weight']
                    Graph.edges[triple[0], edge[1]]['constrained_modularity'] = False
        Graph.remove_node(triple[2])
        
    Graph = nx.convert_node_labels_to_integers(Graph)
    edges_added = []
    ModularityMatrix = get_modularity_matrix(Graph, resolution)
    for i in range(ModularityMatrix.shape[0]):
        for j in range(i, ModularityMatrix.shape[0]):
            if Graph.has_edge(i, j):
                if Graph.edges[(i, j)]['constrained_modularity']:
                    Graph.edges[(i, j)]['weight'] = max(-1, ModularityMatrix[i, j] - 0.5)
                else:
                    Graph.edges[(i, j)]['weight'] = ModularityMatrix[i, j]
            else:
                Graph.add_edge(i, j)
                Graph.edges[(i, j)]['weight'] = ModularityMatrix[i, j]
                Graph.edges[(i, j)]['constrained_modularity'] = False
                edges_added.append((i, j))
    return Graph, edges_added
    
def alter_modularity(G, triple, orig_g, delta, resolution):
    """
    Alter the modularity associated with nodes in triple by input factor (for right branching).
    Returns the reduced graph and additional edges added for running pycombo
    """
    new_triple = [-1,-1,-1]
    value_zero = orig_g.nodes[triple[0]]['super node of'][0]
    value_one = orig_g.nodes[triple[1]]['super node of'][0]
    value_two = orig_g.nodes[triple[2]]['super node of'][0]
    for node in G.nodes():
        if value_zero in G.nodes[node]["super node of"]:
            new_triple[0] = node
        if value_one in G.nodes[node]["super node of"]:
            new_triple[1] = node
        if value_two in G.nodes[node]["super node of"]:
            new_triple[2] = node
    triple = new_triple
    
    Graph = G.copy()
    
    edges_added = []
    ModularityMatrix = get_modularity_matrix(Graph, resolution)
    for i in range(ModularityMatrix.shape[0]):
        for j in range(i, ModularityMatrix.shape[0]):
            if (i, j) in [(triple[0], triple[1]), (triple[0], triple[2]), (triple[1], triple[2]), (triple[1], triple[0]), (triple[2], triple[0]), (triple[2], triple[1])] and Graph.has_edge(i, j):
                Graph.edges[(i, j)]['weight'] = max(-1, ModularityMatrix[i, j] - delta) #replace with max(-1, orig- 0.5)
                Graph.edges[(i, j)]['constrained_modularity'] = True
            elif Graph.has_edge(i, j):
                Graph.edges[(i, j)]['weight'] = ModularityMatrix[i, j]
                Graph.edges[(i, j)]['constrained_modularity'] = False
            else:
                Graph.add_edge(i, j)
                Graph.edges[(i, j)]['weight'] = ModularityMatrix[i, j]
                Graph.edges[(i, j)]['constrained_modularity'] = False
                edges_added.append((i, j))
    return Graph, edges_added

def remove_extra_edges(Graph, edge_list):
    """
    Removes the additional edges added for running pycombo
    """
    for edge in edge_list:
        Graph.remove_edge(edge[0], edge[1])
    return Graph


def reduced_cost_variable_fixing(model, var_vals, obj_value, lower_bound, Graph, resolution):
    AdjacencyMatrix = nx.to_numpy_matrix(Graph, weight="actual_weight")
    ModularityMatrix = get_modularity_matrix(Graph, resolution)
    new_obj_val = ((obj_value*np.sum(AdjacencyMatrix)) - ModularityMatrix.trace()[0, 0])/2
    new_lower_bound = ((lower_bound*np.sum(AdjacencyMatrix)) - ModularityMatrix.trace()[0, 0])/2
    vars_one = []
    vars_zero = []
    for key in var_vals.keys():
        var = model.getVarByName(key)
        if var_vals[key] == 1:
            if new_obj_val - var.getAttr(GRB.Attr.RC) < new_lower_bound:
                vars_one.append(key)
        elif var_vals[key] == 0:
            if new_obj_val + var.getAttr(GRB.Attr.RC) < new_lower_bound:
                vars_zero.append(key)
    return vars_one, vars_zero


class Node:
    """
    Represents one node in the bayan tree
    """
    def __init__(self, constraint_list, var_vals, g, combo_comms):
        self.constraints = constraint_list
        self.var_vals = var_vals
        self.lower_bound = None
        self.upper_bound = None
        self.graph = g
        self.left = None
        self.right = None
        self.parent = None
        self.close = False
        self.is_integer = False
        self.is_infeasible = False
        self.level = -1
        self.fixed_zeros = []
        self.fixed_ones = []
        for com in combo_comms:
            com.sort()
        combo_comms.sort()
        self.combo_communities = combo_comms
    
    def set_bounds(self, lb, ub):
        self.lower_bound = lb
        self.upper_bound = ub
    
    def get_violated_triples(self, list_of_cut_triads):
        return find_violating_triples(self.graph, self.var_vals, list_of_cut_triads)
    
    def close_node(self):
        self.close = True
    
    def set_is_integer(self):
        self.is_integer = True
    
    def set_is_infeasible(self):
        self.is_infeasible = True
    
    def set_level(self, l):
        self.level = l
        
    def get_constraints(self):
        return self.constraints
    
    def set_fixed_ones(self, ones):
        self.fixed_ones = ones
    
    def set_fixed_zeros(self, zeros):
        self.fixed_zeros = zeros
        
    def get_fixed_ones(self):
        return self.fixed_ones
    
    def get_fixed_zeros(self):
        return self.fixed_zeros


# In[ ]:


def create_bayan_edge_attributes(G, resolution):
    #actual_weight is the attribute that stores the edge weight
    for edge in G.edges():
        G.edges[edge]['constrained_modularity'] = False
        if 'weight' in G.edges[edge]:
            G.edges[edge]['actual_weight'] = G.edges[edge]['weight']
        else:
            G.edges[edge]['actual_weight'] = 1
    
    ModularityMatrix = get_modularity_matrix(G, resolution)
    for edge in G.edges():
        G.edges[edge]['weight'] = ModularityMatrix[int(edge[0]), int(edge[1])]
            
    #weight is the attribute that stores the modularity for pair i, j. This is because pycombo needs the modularity to be the 'weight' attribute
    #super node of stores all the nodes that are a part of this super node
    for node in G.nodes():
        G.nodes[node]['super node of'] = [node]
        
    return G


# In[ ]:


def handle_isolated_nodes(Graph):
    isolated = []
    for x in Graph.nodes():
        if Graph.degree[x] == 0:
            isolated.append(x)
    for x in isolated:
        Graph.remove_node(x)
    Graph = nx.convert_node_labels_to_integers(Graph)
    return Graph, isolated
    


# In[ ]:


def create_int_node_names(Graph):
    G = Graph.copy()
    count = 0
    mapping = {}
    node_name_dict = {}
    for n in G.nodes():
        mapping[n] = count
        node_name_dict[count] = n
        count += 1
    G = nx.relabel_nodes(G, mapping)
    return G, node_name_dict


# In[ ]:


def get_modularity_matrix(Graph, resolution):
    AdjacencyMatrix = nx.to_numpy_matrix(Graph, weight="actual_weight")
    ModularityMatrix = np.empty(AdjacencyMatrix.shape)
    degrees = np.array([x[1] for x in list(Graph.degree(weight="actual_weight"))])
    sub_factor = np.array([AdjacencyMatrix[x, x] for x in range(len(Graph.nodes()))])
    degrees = degrees - sub_factor
    degrees_nx1 = np.reshape(degrees, (degrees.shape[0], 1))
    degrees_1xn = np.reshape(degrees, (1, degrees.shape[0]))
    sum_of_edges = np.sum(AdjacencyMatrix)
    return AdjacencyMatrix - (((degrees_nx1*degrees_1xn)/sum_of_edges)*resolution)    


# In[ ]:


def comms_to_original_name(partition, node_name_dict):
    new_p = []
    for comm in partition:
        new_c = []
        for c in comm:
            new_c.append(node_name_dict[c])
        new_p.append(new_c)
    return new_p

def percentage_diff(v1, v2):
    return str(np.round((abs(v1 - v2)/((v1 + v2)/2))*100, 3)) + "%"

def bayan(G, threshold=0.001, time_allowed=60, delta=0.5, resolution=1):
    """
    Run bayan on input Graph while MIP gap > threshold and runtime < time_allowed (default is 60 seconds)
    """
    print("Disregard this message if you have already registered a free academic license for Gurobi on this machine. Otherwise, follow the instructions on https://github.com/saref/bayan/blob/main/README.md to register a license which then allows you to run the Bayan algorithm on larger networks.")
    print("\n")
    run_start = time.time()
    preprocessing_time_start = time.time()
    G, node_name_dict = create_int_node_names(G)
    G1 = G.copy()
    orig_graph = create_bayan_edge_attributes(G1, resolution)
    G2 = orig_graph.copy()
    Graph, isolated_nodes = handle_isolated_nodes(G2)
    Graph = clique_filtering(Graph, resolution)
    AdjacencyMatrix = nx.to_numpy_matrix(Graph, weight="actual_weight")
    ModularityMatrix = get_modularity_matrix(Graph, resolution)
    size = int(Graph.size(weight='actual_weight'))
    order = len(AdjacencyMatrix)
    preprocessing_time = time.time() - preprocessing_time_start
    mod_lp, var_vals, model, list_of_cut_triads, formulation_time, root_lp_time = lp_formulation(Graph, AdjacencyMatrix, ModularityMatrix, size, order, isolated_nodes)
    if is_integer_solution(Graph, var_vals):
        print("Finished!")
        partition = decluster_communities(model_to_communities(var_vals, Graph), Graph, isolated_nodes)
        return mod_lp, percentage_diff(mod_lp, mod_lp), comms_to_original_name(partition, node_name_dict)
    root_combo_time_start = time.time()
    partition_combo = algorithms.pycombo(Graph, weight="actual_weight")
    communities_combo = list(partition_combo.communities)
    communities_combo_declustered = decluster_communities(communities_combo, Graph, isolated_nodes)
    mod_combo = calculate_modularity(communities_combo_declustered, orig_graph, resolution)
    root_combo_time = time.time() - root_combo_time_start
    if mod_lp - mod_combo < threshold:
        print("Finished!")
        return mod_combo, percentage_diff(mod_lp, mod_combo), comms_to_original_name(communities_combo_declustered, node_name_dict)
    best_bound = mod_lp
    incumbent = mod_combo
    root = Node([], var_vals, Graph, communities_combo_declustered)
    root.set_level(0)
    root.set_bounds(mod_combo, mod_lp)
    var_fixed_ones, var_fixed_zeros = reduced_cost_variable_fixing(model, var_vals, mod_lp, incumbent, Graph, resolution)
    root.set_fixed_ones(var_fixed_ones)
    root.set_fixed_zeros(var_fixed_zeros)
    current_node = root
    current_level = 1
    nodes_previous_level = [root]
    best_combo = root
    best_lp = root
    root_time = root_lp_time + root_combo_time
    solve_start = time.time()
    print("Please wait, Bayan is searching for an optimal solution...")
    while (best_bound - incumbent > threshold and nodes_previous_level != [] and time.time() - solve_start - root_time <= time_allowed): #add time_limit as a user parameter
        nodes_current_level = []
        lower_bounds = []
        upper_bounds = []
        for node in nodes_previous_level:
            if time.time() - solve_start - root_time >= time_allowed:
                if best_combo.is_integer:
                    if best_combo.lower_bound <= best_combo.upper_bound:
                        print("Finished!")
                        partition = decluster_communities(model_to_communities(best_combo.var_vals, Graph), Graph, isolated_nodes)
                        return best_combo.upper_bound, percentage_diff(best_combo.upper_bound, best_combo.upper_bound), comms_to_original_name(partition, node_name_dict)
                else:
                    print("Finished!")
                    return best_combo.lower_bound, percentage_diff(best_lp.upper_bound, best_combo.lower_bound), comms_to_original_name(best_combo.combo_communities, node_name_dict)
            current_node = node
            left_node, right_node = perform_branch(node, model, incumbent, best_bound, Graph, orig_graph, isolated_nodes, list_of_cut_triads, delta, resolution)
            if left_node.close:
                if left_node.is_integer and incumbent <= left_node.upper_bound:
                    incumbent = left_node.upper_bound
                    best_combo = left_node
                    nodes_current_level.append(left_node)
                    lower_bounds.append(left_node.lower_bound)
                    upper_bounds.append(left_node.upper_bound)
                    current_node.left = left_node
                    left_node.parent = current_node
                    left_node.set_level(current_level)
                current_node.left = left_node
            else:
                nodes_current_level.append(left_node)
                lower_bounds.append(left_node.lower_bound)
                upper_bounds.append(left_node.upper_bound)
                current_node.left = left_node
                left_node.parent = current_node
                left_node.set_level(current_level)

            if right_node.close:
                if right_node.is_integer and incumbent <= right_node.upper_bound:
                    incumbent = right_node.upper_bound
                    best_combo = right_node
                    nodes_current_level.append(right_node)
                    lower_bounds.append(right_node.lower_bound)
                    upper_bounds.append(right_node.upper_bound)
                    current_node.right = right_node
                    right_node.parent = current_node
                    right_node.set_level(current_level)
                current_node.right = right_node
            else:
                nodes_current_level.append(right_node)
                lower_bounds.append(right_node.lower_bound)
                upper_bounds.append(right_node.upper_bound)
                current_node.right = right_node
                right_node.parent = current_node
                right_node.set_level(current_level)
        incumbent = max(lower_bounds + [incumbent])
        not_possible_vals = []
        count = 0
        for n in nodes_current_level:
            if n.lower_bound == incumbent:
                best_combo = n
            if n.upper_bound < incumbent:
                n.close = True
                not_possible_vals.append(n.upper_bound)
            count += 1
        for val in not_possible_vals:
            upper_bounds.remove(val)
        best_bound = min(upper_bounds + [best_bound])
        for n in nodes_current_level:
            if n.upper_bound == best_bound:
                best_lp = n
        current_level += 1
        nodes_p_level = []
        for a in nodes_current_level:
            if a.close == False:
                nodes_p_level.append(a)
        nodes_previous_level = nodes_p_level
        
    if best_combo.is_integer:
        if best_combo.lower_bound <= best_combo.upper_bound:
            print("Finished!")
            partition = decluster_communities(model_to_communities(best_combo.var_vals, Graph), Graph, isolated_nodes)
            return best_combo.upper_bound, percentage_diff(best_combo.upper_bound, best_combo.upper_bound), comms_to_original_name(partition, node_name_dict)
    else:
        print("Finished!")
        return best_combo.lower_bound, best_lp.upper_bound, percentage_diff(best_lp.upper_bound, best_combo.lower_bound), comms_to_original_name(best_combo.combo_communities, node_name_dict)


def perform_branch(node, model, incumbent, best_bound, Graph, original_graph, isolated_nodes, list_of_cut_triads, delta, resolution):
    """
    Perform the left and right branch on input node
    """
    violated_triples_dict = node.get_violated_triples(list_of_cut_triads)
    prev_fixed_ones = node.get_fixed_ones().copy()
    prev_fixed_zeros = node.get_fixed_zeros().copy()
    #Select triple based on most common nodes with previous triple
    branch_triple = get_best_triple(violated_triples_dict, node, Graph)
#     print("======== BRANCHING ON " + str(branch_triple) + " ========")
    x_ij = model.getVarByName(str(branch_triple[0]) + "," + str(branch_triple[1]))
    x_jk = model.getVarByName(str(branch_triple[1]) + "," + str(branch_triple[2]))
    x_ik = model.getVarByName(str(branch_triple[0]) + "," + str(branch_triple[2]))
    
    #Left branch x_ij + x_jk + x_ik = 0
    count = 0
    if node.constraints == []:
        model.addConstr(x_ij + x_jk + x_ik == 0, 'branch_0')
        count += 1
    else:
        model.addConstr(x_ij + x_jk + x_ik == 0, 'branch_0')
        count += 1
        for constr in node.constraints:
            x_ij = model.getVarByName(str(constr[0]) + "," + str(constr[1]))
            x_jk = model.getVarByName(str(constr[1]) + "," + str(constr[2]))
            x_ik = model.getVarByName(str(constr[0]) + "," + str(constr[2]))
            if constr[3] == 0:
                model.addConstr(x_ij + x_jk + x_ik == 0, 'branch_' + str(count))
            else:
                model.addConstr(x_ij + x_jk + x_ik >= 2, 'branch_' + str(count))
            count += 1
    model.update()
            
    left_upper_bound, left_var_vals, model = run_lp(model, Graph, prev_fixed_ones, prev_fixed_zeros, resolution)
    if not (left_upper_bound == -1 and left_var_vals == -1):
        left_fix_ones, left_fix_zeros = reduced_cost_variable_fixing(model, left_var_vals, left_upper_bound, incumbent, Graph, resolution)
    model = reset_model_varaibles(model, prev_fixed_ones, prev_fixed_zeros)
    
    for i in range(count):
        model.remove(model.getConstrByName('branch_' + str(i)))
    model.update()
    
    left_graph, edges_added = reduce_triple(node.graph, branch_triple, Graph, resolution)
    left_partition_combo = algorithms.pycombo(left_graph, treat_as_modularity=True)
    left_communities_combo = list(left_partition_combo.communities)
    left_decluster_combo = decluster_communities(left_communities_combo, left_graph, isolated_nodes)
    left_graph = remove_extra_edges(left_graph, edges_added)
    left_lower_bound = calculate_modularity(left_decluster_combo, original_graph, resolution)
#     print(left_lower_bound)
    left_constraints = node.constraints.copy()
    left_constraints.append(branch_triple + (0,))
    left_node = Node(left_constraints, left_var_vals, left_graph, left_decluster_combo)
    left_node.set_bounds(left_lower_bound, left_upper_bound)
    
    
    if left_upper_bound == -1 and left_var_vals == -1:
        left_node.close_node()
        left_node.set_is_infeasible()
    else:
        left_node.set_fixed_ones(prev_fixed_ones + left_fix_ones)
        left_node.set_fixed_zeros(prev_fixed_zeros + left_fix_zeros)
        if is_integer_solution(left_graph, left_var_vals):
            left_node.set_is_integer()
#             left_node.set_bounds(left_upper_bound, left_upper_bound)
            left_node.close_node()
        if left_upper_bound <= incumbent:
            left_node.close_node()
            
    
    #Right branch x_ij + x_jk + x_ik >= 2
    x_ij = model.getVarByName(str(branch_triple[0]) + "," + str(branch_triple[1]))
    x_jk = model.getVarByName(str(branch_triple[1]) + "," + str(branch_triple[2]))
    x_ik = model.getVarByName(str(branch_triple[0]) + "," + str(branch_triple[2]))
    count = 0
    if node.constraints == []:
        model.addConstr(x_ij + x_jk + x_ik >= 2, 'branch_0')
        count += 1
    else:
        model.addConstr(x_ij + x_jk + x_ik >= 2, 'branch_0')
        count += 1
        for constr in node.constraints:
            x_ij = model.getVarByName(str(constr[0]) + "," + str(constr[1]))
            x_jk = model.getVarByName(str(constr[1]) + "," + str(constr[2]))
            x_ik = model.getVarByName(str(constr[0]) + "," + str(constr[2]))
            if constr[3] == 0:
                model.addConstr(x_ij + x_jk + x_ik == 0, 'branch_' + str(count))
            else:
                model.addConstr(x_ij + x_jk + x_ik >= 2, 'branch_' + str(count))
            count += 1
    model.update()
    
    right_upper_bound, right_var_vals, model = run_lp(model, Graph, prev_fixed_ones, prev_fixed_zeros, resolution)
    
    if not (right_upper_bound == -1 and right_var_vals == -1):
        right_fix_ones, right_fix_zeros = reduced_cost_variable_fixing(model, right_var_vals, right_upper_bound, incumbent, Graph, resolution)
    model = reset_model_varaibles(model, prev_fixed_ones, prev_fixed_zeros)
    
    for i in range(count):
        model.remove(model.getConstrByName('branch_' + str(i)))
    model.update()
    
    right_graph, edges_added = alter_modularity(node.graph, branch_triple, Graph, delta, resolution)
    right_partition_combo = algorithms.pycombo(right_graph, treat_as_modularity=True)
    right_communities_combo = list(right_partition_combo.communities)
    right_decluster_combo = decluster_communities(right_communities_combo, right_graph, isolated_nodes)
    right_lower_bound = calculate_modularity(right_decluster_combo, original_graph, resolution)
    right_graph = remove_extra_edges(right_graph, edges_added)
#     print(right_lower_bound)
    right_constraints = node.constraints.copy()
    right_constraints.append(branch_triple + (2,))
    right_node = Node(right_constraints, right_var_vals, right_graph, right_decluster_combo)
    right_node.set_bounds(right_lower_bound, right_upper_bound)
    

    
    if right_upper_bound == -1 and right_var_vals == -1:
        right_node.close_node()
        right_node.set_is_infeasible()
    else:
        right_node.set_fixed_ones(prev_fixed_ones + right_fix_ones)
        right_node.set_fixed_zeros(prev_fixed_zeros + right_fix_zeros)
        if is_integer_solution(right_graph, right_var_vals):
            right_node.set_is_integer()
#             right_node.set_bounds(right_upper_bound, right_upper_bound)
            right_node.close_node()
        if right_upper_bound <= incumbent:
            right_node.close_node()
    return left_node, right_node

