# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:59:40 2018

@author: rosie
"""

import random
from gurobipy import *
from pprint import pprint
from math import ceil, floor

n_i = 7
n_d = 7
n_a = 6
w_i = 3
w_d = 1
c = [13 for i in range(3)]
times = [5, 5, 10]
H = range(len(c))

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
CLOSE_ENOUGH = 1.005

#global map storing the resource cost of any given schedule
# schedule_tuple -> W
schedule_resources = {}

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

def exceeds_threshold(w, q):
    return all(w[i] >= q[i] for i in range(len(w)))

######################################################################
#        SUB-PROBLEM
######################################################################

# TODO: Make T T_j, etc. 
def F(W_, R, i, j, T, s):
    
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

        # Check for branching constraint stuff here
        # Adjust reduced cost if so
        for theta, alpha, q in BranchConstraints:
            if s in theta and exceeds_threshold(W_, q):
                print('Stuff is happening')
                R_j -= BranchConstraints[(theta, alpha, q)].pi
                print('R_j', R_j)

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
    global schedule_resources

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
            W_j, R_j, T_j = F(W_i, R_i, i, j, T_i, s)

            if not any(res > res_max for  res, res_max in zip(W_j, n + c)):
                if not is_dominated_by_set(W_j, R_j, E[j]):
                    if R_j > EPSILON:
                        if j != FINAL_NODE:
                            s_j = s + (vertices[j],)
                        else: 
                            s_j = s

                        # update the global dictionary of schedule costs
                        schedule_resources[s_j] = W_j

                        label = (j, W_j, R_j, T_j, s_j)
                        L.append(label)
                        E[j] = EFF(W_j, R_j, E[j])
                        E[j].add(label)
        #print("Length: ",len(L),", E: ",len(list(E[FINAL_NODE])))
    # Good boi labels
   # pprint(E[FINAL_NODE])

    return max(E[FINAL_NODE], default=None, key=lambda x: x[2])

def solve_RMP():
    global MaxPeople
    global ResourceCapacity
    global MaxAmbulances

    found = 0

    while True:

        for s in lambda_s:
            print(str(s) + " Lambda: " + str(lambda_s[s].x))

        solution = subproblem()

        if solution is None:
            break
        found += 1

        print('#' * 80)
        pprint(solution)
        ls = solution[-1]
        print('Calculated RC', get_gs(ls)-sum(get_people(ls)[p]*MaxPeople[p].pi for p in P)-
              sum(get_resources_used(ls)[h]*ResourceCapacity[h].pi for h in H)-
              MaxAmbulances.pi)
        # pprint(schedules)
    
        # paranoia
        if ls in lambda_s:
            raise ValueError("DUPLICATE VARIABLE:", ls)

        lambda_s[ls] = master.addVar()

        #TODO Change this business to not delete all constraints
        for key in MaxPeople:
            master.remove(MaxPeople[key])
        for key in ResourceCapacity:
            master.remove(ResourceCapacity[key])
        master.remove(MaxAmbulances)

        # new constraints
        MaxPeople = {p: master.addConstr(
            quicksum(get_people(s)[p] * lambda_s[s]
                     for s in lambda_s) <= n[p])
            for p in P}
        ResourceCapacity = {h: master.addConstr(
            quicksum(get_resources_used(s)[h] * lambda_s[s]
                     for s in lambda_s) <= c[h])
            for h in H}
        MaxAmbulances = master.addConstr(quicksum(lambda_s[s] for s in lambda_s) <= n_a)

        # new objective
        master.setObjective(quicksum(get_gs(s) * lambda_s[s] for s in lambda_s), GRB.MAXIMIZE)
        master.optimize()
    print('Solutions found', found)

def solve_RIP():
    for s in lambda_s:
        lambda_s[s].vType = GRB.INTEGER
    master.optimize()

###############################################################################
#       Branch and Price to get Integer Solution
###############################################################################
def is_integer(num):
    return abs(num % 1 - 0.5) > 0.5 - EPSILON

def continue_branching():
    global bestSoFar
    # APPROXIMATION PRUNE
    # if master.objVal < bestSoFar * CLOSE_ENOUGH:
    #     print('Node pruned by approximation')
    #     return False

    if any(not(is_integer(lambda_s[s].x)) for s in lambda_s):
        return True
    if master.objVal > bestSoFar:
        bestSoFar = master.objVal
        print('*******New Incumbent Solution', bestSoFar)
    return False

def solve_node(node):

    global BranchConstraints

    #TODO Figure out how the fuck changing constraints works
    # remove all constraints

    for key in BranchConstraints:
        master.remove(BranchConstraints[key])
    BranchConstraints.clear()

    for theta, alpha, q, upper in node:
        if upper:
            BranchConstraints[(theta, alpha, q)] = master.addConstr(quicksum(
                lambda_s[s]
                for s in theta
            ) >= alpha)
        else:
            BranchConstraints[(theta, alpha, q)] = master.addConstr(quicksum(
                lambda_s[s]
                for s in theta
            ) <= alpha)

    master.optimize()
    solve_RMP()

def determine_node_data(schedule_set):

    # calculate set of non-integer lambdas
    fractional_schedules = tuple(s for s in schedule_set if not is_integer(lambda_s[s].x))
    fractional_costs = [schedule_resources[s] for s in fractional_schedules]

    # loop through em, looking for first undominated schedule
    undominated = 0
    checking = 1
    while checking < len(fractional_costs):
        if is_dominated(fractional_costs[checking], 0, fractional_costs[undominated], 0):
            undominated = checking
        checking += 1

    q = fractional_costs[undominated]

    costs = [schedule_resources[s] for s in schedule_set]

    theta = tuple(schedule_set[i]
                  for i in range(len(schedule_set))
                  if exceeds_threshold(costs[i], q))

    alpha = sum(lambda_s[s].x for s in theta)

    return theta, alpha, q

def generate_priority_schedules():
    pass

######################################################################
#              START
######################################################################

#TODO Delete this
for i in range(3):
    print()

# schedules = all_schedules([], c, n_i, n_d)
schedules = [((0, 0),)]

for s in schedules:
    schedule_resources[s] = tuple(get_people(s)) + tuple(get_resources_used(s))

master = Model('Master Problem')
master.setParam('OutputFlag', 0)

# Variables
lambda_s = {s: master.addVar()  # vtype=GRB.BINARY
            for s in schedules}

# Objective
master.setObjective(quicksum(get_gs(s) * lambda_s[s] for s in lambda_s), GRB.MAXIMIZE)

# Constraints
MaxPeople = {p: master.addConstr(
    quicksum(get_people(s)[p] * lambda_s[s]
             for s in lambda_s) <= n[p])
    for p in P}
ResourceCapacity = {h: master.addConstr(
    quicksum(get_resources_used(s)[h] * lambda_s[s]
             for s in lambda_s) <= c[h])
    for h in H}
MaxAmbulances = master.addConstr(quicksum(lambda_s[s] for s in lambda_s) <= n_a)

BranchConstraints = {}

# for s in lambda_s:
#     if lambda_s[s].x==1:
#         print(schedules[s])
#         print(get_gs(schedules[s]))
master.optimize()


# to prevent gutter trash first incumbent solution
solve_RMP()

solve_RIP()

bestSoFar = master.objVal
print('First RIP Solution', bestSoFar)

for s in lambda_s:
    lambda_s[s].vType = GRB.CONTINUOUS

master.optimize()
solve_RMP()

node_stack = []
node = [] #TODO
nodes_explored = 1


while True:
# test = 1
# while test < 10:
#     test += 1
    if continue_branching():
        theta_q_j, alpha_j, q_j = determine_node_data(tuple(lambda_s.keys()))

        # add on the new tuple to the new branches
        node_true = node + [(theta_q_j, ceil(alpha_j), q_j, True)]
        node_false = node + [(theta_q_j, floor(alpha_j), q_j, False)]

        node_stack.append(node_true)
        node_stack.append(node_false)


    try:
        node = node_stack.pop()
        nodes_explored += 1
        print('picked new node')
    except IndexError:
        break

    print('STACK LENGTH', len(node_stack))
    pprint(node_stack)
    print('SOLVING NODE')
    pprint(node)
    try:
        solve_node(node)
    except ValueError as e:
        print('ERROR:', e)
        print('EXITING...')
        break

#el donzoes
# for s in lambda_s:
#     print(str(s) + " Lambda: " + str(lambda_s[s].x))

print('Schedules selected:')
for s in lambda_s:
    if lambda_s[s].x > EPSILON:
        print(f'{lambda_s[s].x} lot(s) of', s)


print(f'Explored {nodes_explored} nodes')


