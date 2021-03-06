# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:59:40 2018

@author: rosie
"""

import random
import time

from gurobipy import *
from pprint import pprint
from math import ceil, floor

I = 0
D = 1

MAX_NODES = 100

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

EPSILON = 10**-5
CLOSE_ENOUGH = 1.005

# n_i = 5
# n_d = 5
# n_a = 6
# w_i = 3
# w_d = 1
# c = [10 for i in range(3)]
# times = [5, 5, 10]
# H = range(len(c))
#
# n = [n_i, n_d]
# P = [I, D]
# w = [w_i, w_d]
#
# vertices = [(p,h) for p in P for h in H] + [(FINAL_PATIENT, FINAL_HOSPITAL)]
# V = range(len(vertices))
# FINAL_NODE = len(V) - 1

###########################################################################

# TEST_CASES = [f'{scen}_{i}' for scen in ('OPT','MOD','PES') for i in range(50)]
# TEST_CASES = ['_test']
# TEST_CASES = ['PES_0']
TEST_CASES = [f'{scen}_{i}' for scen in ('OPT','MOD','PES') for i in range(5,10)]


###########################################################################

#global map storing the resource cost of any given schedule
# schedule_tuple -> W
schedule_resources = {}

def f_p(t, p):
    """
    :param t: time
    :param p: type of patient (I: immediate, D: delayed)
    :return:
    """
    b_0, b_1, b_2 = SURVIVAL_RATES[SCENARIO][I] if p == I else SURVIVAL_RATES[SCENARIO][D]
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
        for theta, alpha, q, upper in BranchConstraints:
            if s in theta:
                print('Stuff is happening')
                R_j -= BranchConstraints[(theta, alpha, q, upper)].pi
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

                        if j == FINAL_NODE and len(E[j]) > 30:
                            print('Max schedules generated')
                            return [label[-1] for label in E[FINAL_NODE]]

        #print("Length: ",len(L),", E: ",len(list(E[FINAL_NODE])))
    # Good boi labels
   # pprint(E[FINAL_NODE])

    # return max(E[FINAL_NODE], default=None, key=lambda x: x[2])
    return [label[-1] for label in E[FINAL_NODE]]

def solve_RMP():
    global MaxPeople
    global ResourceCapacity
    global MaxAmbulances

    found = 0

    while True:

        #print(lambda_s)
        # for s in lambda_s:
        #     print(str(s) + " Lambda: " + str(lambda_s[s].x))

        # solutions is the list of new schedules
        solutions = subproblem()

        if len(solutions) == 0:
            break
        found += len(solutions)

        # print('#' * 80)
        # pprint(solutions)
        # print('Calculated RC', get_gs(ls)-sum(get_people(ls)[p]*MaxPeople[p].pi for p in P)-
        #       sum(get_resources_used(ls)[h]*ResourceCapacity[h].pi for h in H)-
        #       MaxAmbulances.pi)
        # pprint(schedules)

        # paranoia
        for ls in solutions:
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
    global bestSolution
    global nodes_since_change
    # APPROXIMATION PRUNE
    # if master.objVal < bestSoFar * CLOSE_ENOUGH:
    #     print('Node pruned by approximation')
    #     return False

    print('INTEGER SOLUTION', master.objVal, bestSoFar)
    if master.objVal > bestSoFar and all(is_integer(lambda_s[s].x) for s in lambda_s):
        bestSoFar = master.objVal
        print('*******New Incumbent Solution', bestSoFar)
        nodes_since_change = 0
        bestSolution = {}
        for s in lambda_s:
            if lambda_s[s].x > EPSILON:
                bestSolution[s] = lambda_s[s].x

    if all(is_integer(lambda_s[s].x) for s in lambda_s):
        return False

    if master.objVal < bestSoFar:
        return False

    return True

def set_branch_constraints(node):

    # check for duplicate branch constraints
    for i in range(len(node)):
        for j in range(i + 1, len(node)):
            if node[i] == node[j]:
                # pprint(node[i])
                print('DUPLICATE BRANCH', i, j)
                print(len(node))
                xxxxxx

    # TODO Figure out how the fuck changing constraints works
    # remove all constraints
    for key in BranchConstraints:
        master.remove(BranchConstraints[key])
    BranchConstraints.clear()

    for theta, alpha, q, upper in node:
        if upper:
            BranchConstraints[(theta, alpha, q, upper)] = master.addConstr(quicksum(
                lambda_s[s]
                for s in theta
            ) >= alpha)
        else:
            BranchConstraints[(theta, alpha, q, upper)] = master.addConstr(quicksum(
                lambda_s[s]
                for s in theta
            ) <= alpha)
        print(f'Branch constraint: {upper}, {alpha}, {len(theta)}')
        # pprint(theta)

def solve_node(node):

    set_branch_constraints(node)
    master.optimize()

    for c in master.getConstrs():
        print('CONSTRAINT:', c)
    status = master.status
    if status == GRB.Status.INFEASIBLE or status == GRB.Status.INF_OR_UNBD:
        print('INFEASIBLE')
        return False
    solve_RMP()
    return True

def determine_node_data(schedule_set):

    for s in lambda_s:
        if lambda_s[s].x > 0.5:
            print('SUPER BOOBS', lambda_s[s].x)

    print('Just optimised with:')
    for c in master.getConstrs():
        print(c)

    for k in BranchConstraints:
        print(k)
    print('='*50)
    # calculate set of non-integer lambdas
    fractional_schedules = tuple(s for s in schedule_set if not is_integer(lambda_s[s].x))
    fractional_costs = [schedule_resources[s] for s in fractional_schedules]

    # loop through em, looking for first undominated schedule
    undominated = 0
    checking = 1
    while checking < len(fractional_costs):
        if all(fractional_costs[checking][i] >= fractional_costs[undominated][i]
            for i in range(len(fractional_costs[checking]))):
            undominated = checking
        checking += 1
    print('frac', fractional_schedules, fractional_costs)
    q = fractional_costs[undominated]

    # If we find identical resoure vector in 2 places, branch so that 1 variable
    for i in range(len(fractional_costs)):
        for j in range(i + 1, len(fractional_costs)):
            if fractional_costs[i] == fractional_costs[j]:
                print(f'Duplicate resource vectors {fractional_costs[i]} <-> {fractional_costs[j]}')
                BranchConstraints[((fractional_schedules[i],), 0, (), True)] = master.addConstr(
                    lambda_s[fractional_schedules[i]] == 0)

    print('q:', q)
    print('Non-integer?', lambda_s[fractional_schedules[undominated]].x)

    costs = [schedule_resources[s] for s in schedule_set]

    # for i in range(len(schedule_set)):
    #     if exceeds_threshold(costs[i],q):
    #         print('---', lambda_s[schedule_set[i]].x, costs[i])
    #         if not is_integer(lambda_s[schedule_set[i]].x) and schedule_set[i] != fractional_schedules[undominated]:
                # print('buthole', q)
                # xxxx

    theta = tuple(schedule_set[i]
                  for i in range(len(schedule_set))
                  if exceeds_threshold(costs[i], q))
    # check = 0
    # for i in range(len(schedule_set)):
    #     x = exceeds_threshold(costs[i], q)
    #     # print(f'{i} EXCEEDS: {costs[i]} {q} {x}')
    #     if x:
    #         check += 1
    # print(len(theta))
    # print(check)
    # pprint(fractional_costs)
    # print(fractional_costs[undominated])

    alpha = sum(lambda_s[s].x for s in theta)

    return theta, alpha, q

# Schedule Generation
def add_one_trip(schedules, priority):
    next_schedules = []
    for s in schedules:
        for h in H:
            next_schedules.append(s + ((priority, h),))
    return next_schedules

def generate_priority_schedules(priority, length):
    schedules = [((priority, h),) for h in H]

    next_layer = schedules[:]
    for i in range(length):
        next_layer = add_one_trip(next_layer, priority)
        schedules.extend(next_layer)

    opposite = (priority + 1) % 2
    for s in schedules[:]:
        if len(s) < length:
            schedules.extend([s + x for x in generate_priority_schedules(opposite, length - len(s) - 1)])

    return schedules


######################################################################
#              START
######################################################################

for test_case in TEST_CASES:

    print(f'Running Test Case {test_case}...')
    print('-'*50)
    # initialise values from test cases
    with open(f'test_cases/{test_case}.txt') as data:
        n_i = int(data.readline().split('#')[0])
        n_d = int(data.readline().split('#')[0])
        n_a = int(data.readline().split('#')[0])
        w_i = int(data.readline().split('#')[0])
        w_d = int(data.readline().split('#')[0])
        c = [int(h) for h in data.readline().split('#')[0].split(',')]
        times = [int(t) for t in data.readline().split('#')[0].split(',')]
        SCENARIO = int(data.readline().split('#')[0])

    H = range(len(c))

    n = [n_i, n_d]
    P = [I, D]
    w = [w_i, w_d]

    vertices = [(p, h) for p in P for h in H] + [(FINAL_PATIENT, FINAL_HOSPITAL)]
    V = range(len(vertices))
    FINAL_NODE = len(V) - 1

    start_time = time.time()

    # schedules = all_schedules([], c, n_i, n_d)
    # schedules = [((0, 0),)]
    schedules = generate_priority_schedules(I, (n_i + n_d) // n_a)


    for s in schedules:
        schedule_resources[s] = tuple(get_people(s)) + tuple(get_resources_used(s))

    nodes_since_change = 0

    master = Model('Master Problem')
    master.setParam('OutputFlag', 1)

    # Variables
    lambda_s = {s: master.addVar()
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
    bestSolution = {}
    for s in lambda_s:
        if lambda_s[s].x > EPSILON:
            bestSolution[s] = lambda_s[s].x

    for s in lambda_s:
        lambda_s[s].vType = GRB.CONTINUOUS

    master.optimize()
    solve_RMP()

    node_stack = []
    node = [] #TODO
    nodes_explored = 1
    feasible = True

    while True:
        # set_branch_constraints(node)
        # master.optimize()
        if feasible and continue_branching():
            theta_q_j, alpha_j, q_j = determine_node_data(tuple(lambda_s.keys()))

            print('DETERMINED:', theta_q_j, alpha_j, q_j)

            # add on the new tuple to the new branches
            node_true = node + [(theta_q_j, ceil(alpha_j), q_j, True)]
            node_false = node + [(theta_q_j, floor(alpha_j), q_j, False)]

            node_stack.append(node_true)
            node_stack.append(node_false)

        try:
            node = node_stack.pop()
            nodes_explored += 1
            nodes_since_change += 1

            if nodes_since_change > MAX_NODES:
                print(f'No incumbent change in {MAX_NODES} nodes')
                break

            print('picked new node')
        except IndexError:
            break

        # print('STACK LENGTH', len(node_stack))
        # pprint(node_stack)
        # print('SOLVING NODE')
        # pprint(node)
        try:
            feasible = solve_node(node)
        except ValueError as e:
            print('!' * 70)
            print('ERROR:', e)
            print('EXITING...')
            print('!' * 70)
            break

    #el donzoes
    # for s in lambda_s:
    #     print(str(s) + " Lambda: " + str(lambda_s[s].x))

    duration = time.time() - start_time

    # print('Schedules selected:')
    for s in bestSolution:
        print(f'{round(bestSolution[s], 3)} lot(s) of', s)
    print(f'Successfully ran test case `{test_case}`')
    print('Optimal Value Determined:', bestSoFar)

    print(f'Explored {nodes_explored} nodes')
    print(f'Time taken: {duration} seconds')
    print('#'*50)

    with open(f'test_results/branch_and_price/{test_case}_results.txt', 'w') as test_result:
        for s in bestSolution:
            test_result.write(f'{round(bestSolution[s], 3)} lot(s) of {s}\n')
        test_result.write(f'Optimal Value Determined: {bestSoFar}\n')
        test_result.write(f'Explored {nodes_explored} nodes')
        test_result.write(f'Time taken: {duration} seconds')
