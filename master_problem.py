# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:59:40 2018

@author: rosie
"""

from gurobipy import *
from pprint import pprint
import random

from copy import deepcopy

n_i = 4
n_d = 4
n_a = 10
w_i = 3
w_d = 3
c = [9 for i in range(3)]
times = [10, 10, 5]

I = 0
D = 1
n = [n_i, n_d]
P = [I, D]
w = [w_i, w_d]

PES = 0
MOD = 1
OPT = 2

SURVIVAL_RATES = {
    PES: ((0.09, 17, 1.10), (0.57, 61, 2.03)),
    MOD: ((0.24, 47, 1.30), (0.76, 138, 2.17)),
    OPT: ((0.56, 91, 1.58), (0.81, 160, 2.41))
}

START_PATIENT = -1
START_HOSPITAL = -1
START_NODE = -1

FINAL_PATIENT = 0
FINAL_HOSPITAL = 0
FINAL_NODE = None

vertices = [(p,h) for p in P for h in H] + [(FINAL_PATIENT, FINAL_HOSPITAL)]
V = range(len(vertices))
FINAL_NODE = len(V) - 1

EPSILON = 10**-7

def f_p(t, p):
    """
    :param t: time
    :param p: type of patient (I: immediate, D: delayed)
    :return:
    """
    b_0, b_1, b_2 = SURVIVAL_RATES[PES][I] if p == I else SURVIVAL_RATES[PES][D]
    return b_0 / (((t / b_1) ** b_2) + 1)

def get_resources_used(schedule):
    b_h = [0 for h in range(len(c))]
    for p, h in schedule:
         b_h[h] += w_i if p == I else w_d
    return b_h    
    
    
def get_people(schedule):
    a = [0, 0]
    for p, h in schedule:
        a[p] += 1
    return a

def get_gs(schedule):
    p_i,h_i = schedule[0]
    t = times[h_i]
    g = f_p(t, p_i)
    
    for i in range(1, len(schedule)):
        p_j,h_j = schedule[i]
        
        t += times[h_i] + times[h_j]
        g += f_p(t,p_j)
        
        p_i,h_i = p_j,h_j
    return g

######################################################################
#        SUB-PROBLEM
######################################################################

# TODO: Make T T_j, etc. 
def F(W_, R, i, j, T):
    
    W_ = list(W_)
    # previous trip
    p_i, h_i = vertices[i] if i != START_NODE else (0, 0)
    
    # current vertex
    p_j, h_j = vertices[j] 
    
    if i==START_NODE:
        T = times[h_j]
    elif j != FINAL_NODE:
        T += times[h_i] + times[h_j]
           
    if j==FINAL_NODE: 
        sigma = MaxAmbulances.pi
        R_j = R-sigma

        #TODO Check for branching constraint stuff here
            #TODO Adjust reduced cost if so

        return tuple(W_), R_j, T
        
    else:
        
        W_[p_j] = W_[p_j] + 1
        W_[h_j + 2] = W_[h_j + 2] + w[p_j]
        
        # dual variables
        pi = MaxPeople[p_j].pi
        rho = ResourceCapacity[h_j].pi
        
       # print('IT\s TIME!', pi, rho, sigma)
        R_j = R + f_p(T, p_j) - pi - w[p_j]*rho 
        return tuple(W_), R_j, T
    
    # print('R_j', R_j)

    return tuple(W_), R_j, T

def is_dominated(W, R, W_, R_):
    """returns whether the label W_,R_ is dominated by label W,R"""
    # for i in P:
    #     # people count
    #     if W_ [i] > W[i]:
    #         return False
    # for i in range(2, len(W)):
    for i in range(len(W)):
        # people and hospital resources
        if W_ [i] < W[i]:
            return False
    if R_ > R:
        return False

    return True

def is_dominated_by_set(W_, R_, labels):
    """returns whether the label W_, R_ is dominated by any vector in labels"""
    for i, W, R, t, s in labels:
        if is_dominated(W, R, W_, R_):
            # W_,R_ is dominated by W,R
            return True
    return False

def EFF(W, R, labels):
    """returns the set of labels from labels that is not dominated by W, R"""
    eff = set()
    for i, W_, R_, t, s in labels:
        if not(all(W[i] <= W_[i] for i in range(len(W))) and R >= R_):
            # W_,R_ is not dominated by W,R
            eff.add((i, W_, R_, t, s))
    return eff


def subproblem():
    #Step 0 Initialisation
    T = 0   #current time
    W_i = (0, 0) + tuple([0 for h in H])
    R_i = 0
    s = tuple()

    E = [set() for v in V]
    L = [(START_NODE, W_i, R_i, T, s)]

    while len(L) > 0:
        #Step 1
        i, W_i, R_i, T_i, s = L.pop(0)
        # pprint(len(L))
        
        if i == len(V):
            # node v_f
            continue
        #Step 2
        for j, v in enumerate(vertices):
            W_j, R_j, T_j = F(W_i, R_i, i, j, T_i)

            if not any(res > res_max for  res, res_max in zip(W_j, n + c)):
                if not is_dominated_by_set(W_j, R_j, E[j]):
                    if (R_j > EPSILON):
                        if j != FINAL_NODE:
                            s_j = s + (vertices[j],)
                        else: 
                            s_j = s
                        label = (j, W_j, R_j, T_j, s_j)
                        L.append(label)
                        E[j] = EFF(W_j, R_j, E[j])
                        E[j].add(label)
        #print("Length: ",len(L),", E: ",len(list(E[FINAL_NODE])))
    # Good boi labels
   # pprint(E[FINAL_NODE])

    return max(E[FINAL_NODE], default=None, key=lambda x: x[2])

######################################################################
#              MASTER PROBLEM
#
#              Adding good schedules from subproblem
######################################################################

H = range(len(c))
# schedules = all_schedules([], c, n_i, n_d)
schedules = [[(0,0)]]
S = range(len(schedules))

master = Model('Master Problem')
master.setParam('OutputFlag', 0)

# Variables
lambda_s = {s: master.addVar() #vtype=GRB.BINARY
            for s in S}

# Objective
master.setObjective(quicksum(get_gs(schedules[s]) * lambda_s[s] for s in S), GRB.MAXIMIZE)

# Constraints
MaxPeople = {p: master.addConstr(
                    quicksum(get_people(schedules[s])[p] * lambda_s[s]
                             for s in S) <= n[p])
             for p in P}
ResourceCapacity = {h: master.addConstr(
                        quicksum(get_resources_used(schedules[s])[h] * lambda_s[s]
                                 for s in S) <= c[h])
                    for h in H}
MaxAmbulances = master.addConstr(quicksum(lambda_s[s] for s in S) <= n_a)

BranchConstraints = []

# for s in lambda_s:
#     if lambda_s[s].x==1:
#         print(schedules[s])
#         print(get_gs(schedules[s]))
master.optimize()

def solve_RMP():
    while True:
        
        global MaxPeople
        global ResourceCapacity
        global MaxAmbulances

        solution = subproblem()

        if solution is None:
            break
        # print('#' * 80)
        pprint(solution)
        ls = list(solution[-1])
        print('Calculated RC', get_gs(ls)-sum(get_people(ls)[p]*MaxPeople[p].pi for p in P)-
              sum(get_resources_used(ls)[h]*ResourceCapacity[h].pi for h in H)-
              MaxAmbulances.pi)
        # pprint(schedules)
    
    
        schedules.append(list(solution[-1]))
        S = range(len(schedules))
    
        lambda_s[len(S)-1] = master.addVar()


        #TODO Change this business to not delete all constraints
    
        # new constraints
        MaxPeople = {p: master.addConstr(
            quicksum(get_people(schedules[s])[p] * lambda_s[s]
                     for s in S) <= n[p])
            for p in P}
        ResourceCapacity = {h: master.addConstr(
                            quicksum(get_resources_used(schedules[s])[h] * lambda_s[s] for s in S) <= c[h])
                            for h in H}
        MaxAmbulances = master.addConstr(quicksum(lambda_s[s] for s in S) <= n_a)
    
        # new objective
        master.setObjective(quicksum(get_gs(schedules[s]) * lambda_s[s] for s in S), GRB.MAXIMIZE)
        master.optimize()
        for s in S:
            print(str(schedules[s])+" Lambda: "+str(lambda_s[s].x))



def is_integer(num):
    return abs(num % 1 - 0.5) > 0.5 - EPSILON

def solve_RIP():
    for s in S:
        lambda_s[s].vType = GRB.INTEGER
    master.optimize()

##############################################################################
#           START
##############################################################################

node_stack = []

solve_RMP()
solve_RIP()

BF4EVA = master.objVal

for s in S:
    lambda_s[s].vType = GRB.CONTINUOUS
    

node_stack.append([[], schedules])
###############################################################################
#       Branch and Price to get Integer Solution
###############################################################################

def continue_branching():
    
    if any(not(is_integer(s)) for s in lambda_s):
        return True
    if master.objVal < BF4EVA:
        BF4EVA = master.objVal
        print('*******New Incumbent Solution', BF4EVA)
    return False

def solve_node(node):
    #get parent's schedules for lambda_s variables
    global schedules
    schedules = node[1]

    #get all theta_q_j and alpha_j from parent -> add branch constraints

    #TODO Figure out how the fuck changing constraints works
    # remove all constraints
    master.remove(master.getConstrs())
    BranchConstraints.clear()

    for q, alpha, upper in node[0]:
        if upper:
            BranchConstraints[]
        else:


    solve_RMP()

def determine_new_alpha_j_new_q():
    pass

while True:
    
    if continue_branching():
        alpha, q = determine_new_alpha_j_new_q_new_Theta_q_j()



        node_stack.append([q_j_and_a_js_and_high_low, deepcopy(schedules), True])
        node_stack.append([q_j_and_a_js_and_high_low, deepcopy(schedules), False])
    
    try:
        Theta_q_j_and_a_js_and_high_low, schedules = CurrNode = node_stack.pop()
    except IndexError:
        break
    
    solve_node(CurrNode)
    
    
#el donzoes
print('We did it friends!', master)


