import numpy as np
from causallearn.utils.GraphUtils import GraphUtils
import pydot
from IPython.display import Image, display

from scipy.special import expit, logit

# convention
# in adjacency matrices:
# 2: circle
# 1: arrowhead
#-1: tail
def fscore_calculator_skeleton(adj,true_adj):
    adj=adj.astype(int)
    true_skeleton=(np.abs(true_adj)>0).astype(int)

    _skeleton=(np.abs(adj)>0).astype(int)
    _tp= 0.5*np.sum((_skeleton*true_skeleton)==1)
    _fp=0.5*np.sum(_skeleton)-_tp
    _tn=0.5*np.sum((1-_skeleton)*(1-true_skeleton))
    _fn=0.5*np.sum((1-_skeleton))-_tn
    _precision=_tp/(_tp+_fp)
    _recall=_tp/(_tp+_fn)
    _fscore=2*_precision*_recall/(_precision+_recall)
    return _fscore

def fscore_calculator_arrowhead(adj,true_adj):
    adj=adj.astype(int)
    true_mark=(true_adj==1).astype(int)

    _mark=(adj==1).astype(int)
    _tp= np.sum((_mark*true_mark)==1)
    _fp=np.sum(_mark)-_tp
    _tn=np.sum((1-_mark)*(1-true_mark))
    _fn=np.sum((1-_mark))-_tn
    _precision=_tp/(_tp+_fp)
    _recall=_tp/(_tp+_fn)
    _fscore=2*_precision*_recall/(_precision+_recall)
    return _fscore

def fscore_calculator_tail(adj,true_adj):
    adj=adj.astype(int)
    true_mark=(true_adj==-1).astype(int)

    _mark=(adj==-1).astype(int)
    _tp= np.sum((_mark*true_mark)==1)
    _fp=np.sum(_mark)-_tp
    _tn=np.sum((1-_mark)*(1-true_mark))
    _fn=np.sum((1-_mark))-_tn
    _precision=_tp/(_tp+_fp)
    _recall=_tp/(_tp+_fn)
    _fscore=2*_precision*_recall/(_precision+_recall)
    return _fscore

def visualize_graph(D,name=False,layout=False):
    if name:
        if layout:
            pyd=GraphUtils.to_pydot(D)
            pdy.set_layout('neato')
            pyd.write_pdf(name+'.pdf') 
        else:
            pyd=GraphUtils.to_pydot(D)
            m=pyd.get_layout()
            pyd.write_pdf(name+'.pdf') 
            return m
    else:
        pyd=GraphUtils.to_pydot(D)
        im = Image(pyd.create_jpeg())
        #Display the image
        display(im) 
        
        
# edit CPTs
def create_CPT(bn,var_name,no_of_states_dict,option,vec=None):
    if option=='random':
        bn.generateCPT(var_name)

    elif option=='logistic_binary':
        #print(var_name)
        parent_names=bn.cpt(var_name).var_names
        for j in parent_names:
            assert no_of_states_dict[j]==2, "logistic_binary can only be used with binary variables"
        parent_names.remove(var_name)
        #print(parent_names)
        assert(len(parent_names)+1== len(vec)), "Length of the vector of coefficients mis matched with the number of parents"
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        #print(parent_states)
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            my_dist=[vec[k]*int(j[k]) for k in range(len(parent_names))]
            logit=np.sum(np.array(my_dist))+vec[-1]
            #print(logit)
            bn.cpt(var_name)[my_dict] = np.array([expit(logit),1-expit(logit)])

    elif option=='deterministic':
        alpha=np.zeros((no_of_states_dict[var_name],))
        #print(no_of_states_dict[var_name])
        alpha[1]=1
        #print(alpha)

        parent_names=bn.cpt(var_name).var_names
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            #print(j)
            alpha_shifted=np.roll(alpha,counter)
            #print({k:1 for k in range(len(parent_names))})        
            #print({parent_names[k]:j for k in range(len(parent_names))})
            #print(parent_names)
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            #print(my_dict)
            my_dist=alpha_shifted
            #print(my_dist)
            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1
    elif option=='Dirichlet':
        alpha=np.ones((no_of_states_dict[var_name],))
        #print(no_of_states_dict[var_name])
        #print(alpha)

        parent_names=bn.cpt(var_name).var_names
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            my_dist=list(np.random.dirichlet(tuple(alpha), 1)[0])
            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1
    elif option=='Meek':
        base=1./np.arange(1,no_of_states_dict[var_name]+1)
        base=base/np.sum(base)
        # equivalent sample size = sum of ai's in Dirichlet
        alpha=10*base
        parent_names=bn.cpt(var_name).var_names
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        #print(parent_names)
        #print(parent_states)
        counter=0
        for j in parent_states:
            if not (isinstance(j,list)):
                j=[j]
            #print(j)
            alpha_shifted=np.roll(alpha,counter)
            # alpha_shifted=np.roll(alpha,np.random.choice(no_of_states[i],1))        
            #print({k:1 for k in range(len(parent_names))})        
            #print({parent_names[k]:j for k in range(len(parent_names))})
            #print(parent_names)
            my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
            #print(my_dict)

            my_dist=list(np.random.dirichlet(tuple(alpha_shifted), 1)[0])

            #my_dist=alpha_shifted

            #print(my_dist)
            bn.cpt(var_name)[my_dict] = my_dist
            counter+=1
    elif option=='reverseDeterministic':
        #print(var_name)
        parent_names=bn.cpt(var_name).var_names
        parent_names.remove(var_name)
        #print(parent_names)
        parent_states=Cartesian([list(np.arange(0,no_of_states_dict[j])) for j in parent_names],len(parent_names))
        #print(parent_states)
        M=np.zeros([len(parent_states),no_of_states_dict[var_name]])
        ind=np.random.choice(len(parent_states),no_of_states_dict[var_name])
        for i in range(no_of_states_dict[var_name]):
            M[ind[i],i]=1
            M=M/np.repeat(np.reshape(np.sum(M,1),[-1,1]),no_of_states_dict[var_name],axis=1)
            for j in parent_states:
                if not (isinstance(j,list)):
                    j=[j]
                my_dict={parent_names[k]:int(j[k]) for k in range(len(parent_names))}
                my_dist=M[j,:]
                #print(logit)
                bn.cpt(var_name)[my_dict] = my_dist
                
# source: https://www.geeksforgeeks.org/cartesian-product-of-any-number-of-sets/
def cartesianProduct(set_a, set_b): 
    result =[] 
    for i in range(0, len(set_a)): 
        for j in range(0, len(set_b)): 
  
            # for handling case having cartesian 
            # prodct first time of two sets 
            if type(set_a[i]) != list:          
                set_a[i] = [set_a[i]] 
                  
            # coping all the members 
            # of set_a to temp 
            temp = [num for num in set_a[i]] 
              
            # add member of set_b to  
            # temp to have cartesian product      
            temp.append(set_b[j])              
            result.append(temp)   
              
    return result 
  
# Function to do a cartesian  
# product of N sets  
def Cartesian(list_a, n): 
      
    # result of cartesian product 
    # of all the sets taken two at a time 
    if len(list_a)==0:
        return []
    else:
        temp = list_a[0] 

        # do product of N sets  
        for i in range(1, n): 
            temp = cartesianProduct(temp, list_a[i]) 
    return temp