import os
import sys
import time

os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"

import pydot
from IPython.display import Image, display

sys.path.append("")
import unittest

import numpy as np
import pandas as pd

from causallearn.search.ConstraintBased.FCI import fci

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import chisq, fisherz, kci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph

import pyAgrum as gum
import pyAgrum.lib.notebook as gnb
import pyAgrum.lib.image as gumimage

from kPC import kPC, make_kess_graph

from causallearn.utils.TXT2GeneralGraph import txt2generalgraph

from utils import fscore_calculator_skeleton, fscore_calculator_arrowhead, fscore_calculator_tail
from utils import visualize_graph

from utils import create_CPT

from pyAgrum.lib.bn2graph import BN2dot

def refillCPT_Dirichlet(bn,node_names,domain,option='Dirichlet'):
    no_of_states_dict = {}
    for i in node_names:
        no_of_states_dict[i] = bn.variable(i).domainSize()

    for var_name in node_names:
        create_CPT(bn,var_name,no_of_states_dict,option)

setID=1
dir_name = 'set'+str(setID)
os.mkdir(dir_name)
n = 10
domain=2 # states for each node. 
node_names=['X'+str(i) for i in np.arange(1,n+1)]

#bn=gum.randomBN(n=n) # binary by default. also initializes the cpts
pc_list=[]
kpc_list=[]
N=100
ratio_arc=3
param_file=dir_name+'/'+'params'
np.savez(param_file,n=n,domain=domain,N=N,density=ratio_arc)

for i in range(N):
    bn=gum.randomBN(n=n,names=node_names,ratio_arc=ratio_arc,domain_size=domain) # ratio_arc=1.2 default
    bn.generateCPTs()
    refillCPT_Dirichlet(bn,node_names,domain)    

    bn.saveBIF(dir_name+'/'+str(i)+'.bif')
    
def create_name_dict(bn): # consistently recover nodeID-nodeName mapping. Gives col/row numbers in adj matrix
    name_dict = {}
    for i in np.arange(1,n+1):#bn.names()
        #print('X'+str(i),bn.nodeId(bn.variableFromName('X'+str(i))))
        name_dict[bn.nodeId(bn.variableFromName('X'+str(i)))]=i
    return name_dict

def get_ess_adj(bnEs,n):
    ess_adj = np.zeros((n,n))
    for i in bnEs.arcs():
        ess_adj[i[1],i[0]]=1
        ess_adj[i[0],i[1]]=-1
    for i in bnEs.edges():
        ess_adj[i[1],i[0]]=-1
        ess_adj[i[0],i[1]]=-1
    return ess_adj

def combined_metric(my_list,N,proj=None):
    if proj=='total':
        total=[my_list[i][0]+my_list[i][1]+my_list[i][2] for i in range(N)]
    elif proj=='arrow':
        total=[my_list[i][1] for i in range(N)]
    elif proj=='tail':
        total=[my_list[i][2] for i in range(N)]
    elif proj =='skel':
        total=[my_list[i][0] for i in range(N)]
    elif proj =='arr_tail':
        total=[my_list[i][1]+my_list[i][2] for i in range(N)]
    else:
        print('Error!! Enter projection type for F1 scores!')
    #total=[my_list[i][1] for i in range(N)]
    return total

def PP_graph(adj):
    # remove bidirected edges since they are known to not be present in any DAG
    n = np.shape(adj)[0]
    for i in range(n):
        for j in range(n):
            if adj[i,j]==1 and adj[j,i]==1:
                adj[i,j]=0
                adj[j,i]=0
    return adj

def get_adj(bn,n):
    adj_ = np.zeros((n,n))
    for i in bn.arcs():
        adj_[i[1],i[0]]=1
        adj_[i[0],i[1]]=-1
    return adj_

showgraphs=False

def plot_cdf(my_list,N,proj,line_style,color):
    total=combined_metric(my_list,N,proj)
    new_total=[]
    for i in total:
        if np.isnan(i):
            new_total.append(0)
        else:
            new_total.append(i)
    total=new_total
    H,X1=np.histogram(total,bins=100,density=True)
    dx = X1[1] - X1[0]
    F1 = np.cumsum(H)*dx
    plt.plot(X1[1:], F1,line_style,color=color,linewidth=2)

for NO_SAMPLES in [10,50,100,250,500,1000]:

    alpha = 0.05

    # setID is preserved from prev cell

    dir_name = 'set'+str(setID)
    param_file=dir_name+'/'+'params'
    params=np.load(param_file+'.npz')
    n=int(params['n'])
    domain=int(params['domain'])
    N=int(params['N'])
    density=int(params['density'])
    proj_types=['total','arrow','tail','skel','arr_tail']

    k_range=[0,1,2]

    max_degree_list = []
    pc_list=[]
    kpc_dict = {}
    for k in k_range:
        kpc_dict[k]=[]
    for i in range(N):  
        print('Working on instance %d'%i)
        bn = gum.loadBN(dir_name+'/'+str(i)+'.bif')
        if showgraphs:
            gr = BN2dot(bn)
            gr.write(dir_name+'/'+'test_instance_'+str(i)+'True.pdf', format='pdf')
        for t in range(3): # using 3 datasets to average out finite-sample effects
            g=gum.BNDatabaseGenerator(bn)
            g.drawSamples(NO_SAMPLES)
            df=g.to_pandas()
            data=df.to_numpy()

            tester=chisq

            cg = pc(data, alpha, tester, verbose=False)        

            G=cg.G

            if t==0 and showgraphs:
                visualize_graph(G,name=dir_name+ '/' + 'test_instance_' + str(i)+'PC')
            adj=G.graph

            bnEs=gum.EssentialGraph(bn)

            ### COMPARE AGAINST ESS OR DAG
            ess_adj=get_ess_adj(bnEs,n) 
            #true_adj = get_adj(bn,n)
            #ess_adj = true_adj# use ground truth graph
            
            #print(fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj))
            pc_list.append((fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj)))
            #visualize_graph(G) 
            #bnEs
            #k=round(0.1*n)
            #print(k)
            for k in k_range:
                D,_ = kPC(data,tester,k,n,alpha=alpha)
                if t==0 and showgraphs:
                    visualize_graph(D, name=dir_name+ '/' + 'test_instance_' + str(i)+'k_' + str(k)) 
                adj = D.graph
                # OPTIONAL: Post-processing and removing <-> edges. 
                # With finite-sample noise, this mostly hurts performance due to extra bidirected
                # edges recovered because of incorrect marginal independences
                #adj = PP_graph(adj)

                #print(fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj))
                kpc_dict[k].append((fscore_calculator_skeleton(adj,ess_adj),fscore_calculator_arrowhead(adj,ess_adj),fscore_calculator_tail(adj,ess_adj)))    

    def title_name(proj):
        if proj=='total':
            title='$F_1^{sk}$+$F_1^{ar}$+$F_1^{ta}$'
        elif proj=='arrow':
            title='$F_1^{ar}$'
        elif proj=='tail':
            title='$F_1^{ta}$'
        elif proj =='skel':
            title='$F_1^{sk}$'
        elif proj =='arr_tail':
            title='$F_1^{ar}$+$F_1^{ta}$'
        else:
            print('Error!! Enter projection type for F1 scores!')
        #total=[my_list[i][1] for i in range(N)]
        return title
    import matplotlib.pyplot as plt
    SMALL_SIZE = 20
    MEDIUM_SIZE = 24
    BIGGER_SIZE = 26

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    for proj in proj_types:
        plt.figure()
        my_list = pc_list
        line_style='-'
        color='maroon'
        plot_cdf(my_list,N,proj,line_style,color)

        colors=['cornflowerblue','seagreen','royalblue']
        linetypes=['--','-.',':']
        for k in k_range:
            my_list = kpc_dict[k]
            line_style=linetypes[k]
            color=colors[k]
            plot_cdf(my_list,N,proj,line_style,color)

        title=title_name(proj)
        plt.title('CDF of '+title+', N=%d'%NO_SAMPLES)
        legend_text = ['PC']+['kPC,k='+str(i) for i in k_range]
        #plt.legend(['PC','kPC,k=1', 'kPC,k=2'])
        plt.legend([i for i in legend_text],loc='lower right')
        ax = plt.gca()
        #ax.set_ylim([0, 1])
        #plt.savefig(dir_name+'/'+'cdf_combined_n_%d_k_%d_dom_%d_den_%d_samples_%d.pdf'%(n,k,domain,density,NO_SAMPLES),transparent=True)
        plt.savefig(dir_name+'/'+'cdf_'+proj+'_n_%d_k_%d_dom_%d_den_%d_samples_%d.pdf'%(n,k,domain,density,NO_SAMPLES),transparent=True)
        plt.close()