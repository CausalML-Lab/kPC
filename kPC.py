import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.cit import chisq, fisherz, kci
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint

from FCI_k import fci_k

def kPC(data,tester,k,n,alpha=0.05,verbose=False):
    G, edges = fci_k(data, tester, alpha, depth=k, verbose=verbose)
    adj=G.graph
    new_adj=kPC_orientations(G,n)
    while (new_adj!=adj).any():
        adj=new_adj
        D=make_kess_graph(new_adj,n)
        new_adj = kPC_orientations(D,n)
    D=make_kess_graph(new_adj,n)
    return D,new_adj

def kPC_orientations(G,n):
    n=G.graph.shape[0]
    adj = G.graph
    #################################

    # Find nodes with no incoming edges:
    # parse rows
    candidate_nodes=[] 
    for i in range(n):
        if (adj[i,:]==1).any():
            pass
        else:
            candidate_nodes.append(i)
    print(candidate_nodes)

    # Now for each one of those nodes, find B and C sets:
    B={}
    for my_node in candidate_nodes:
        B[my_node]=[]
        cand=np.where(adj[my_node,:]==2)[0]
        for i in cand:
            if adj[i,my_node]== 1:#check if the other side of the edge is an arrowhead
                B[my_node].append(i)
    print(B)

    C={}
    for my_node in candidate_nodes:
        C[my_node]=[]
        cand=np.where(adj[my_node,:]==2)[0]
        for i in cand:
            if adj[i,my_node]== 2:#check if the other side of the edge is a circle
                C[my_node].append(i)
    print(C)

    # C* is subset of C that only contains nodes that are non-adjacent to the other nodes in C
    C_star = {}
    for my_node in candidate_nodes:
        C_star[my_node]=[]    
        C_my_node = C[my_node]
        for i in C_my_node: # a circle neighbor to my_node
            neigh=np.where(adj[i,:]!=0)[0] # circle neighbors' neighbors w/ any orientation
            flag=1
            for j in neigh:
                if (j!=my_node):
                    if np.any(C_my_node==j):
                        flag=0
            if flag==1:
                C_star[my_node].append(i)
    print(C_star)

    # B* is subset of B that only contains nodes that are non-adjacent to any of the nodes in C
    B_star = {}
    for my_node in candidate_nodes:
        B_star[my_node]=[]    
        B_my_node = B[my_node]
        C_my_node = C[my_node]

        for i in B_my_node: # a o--> neighbor to my_node
            neigh=np.where(adj[i,:]!=0)[0] # o--> neighbors' neighbors w/ any orientation
            flag=1
            for j in neigh:
                if (j!=my_node):
                    if np.any(C_my_node==j):
                        flag=0
            if flag==1:
                B_star[my_node].append(i)
    print(B_star)

    # re-orient them
    new_adj=adj
    for node in candidate_nodes:
        for i in B_star[node]:
            new_adj[node,i]=-1
        for i in C_star[node]:
            new_adj[node,i]=-1
            new_adj[i,node]=-1
    return new_adj  
#######################

def make_kess_graph(new_adj,n):
    nodes=[]
    for i in np.arange(1,n+1):
        nodes.append(GraphNode('X'+str(i)))
    D = GeneralGraph(nodes)    
    for i in range(n):
        #neigh=new_adj[i,:]
        for j in range(n):
            if new_adj[i,j]==0:
                pass
            else:
                D.add_edge(Edge(nodes[i],nodes[j],Endpoint(new_adj[i,j]),Endpoint(new_adj[j,i])))
    D.pag=True
    return D