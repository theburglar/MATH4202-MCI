from gurobipy import *

n_i = 3
n_d = 4
n_a = 5
w_i = 3
w_d = 3
c_h = [10 for i in range(5)]

I = 0
D = 1
n = [n_i, n_d]
P = [I, D]

H = []
l = []
S = []
g_s = 0

master = Model('Master Problem')

# Variables
lambda_s = {s: master.addVar(vtype=GRB.BINARY)
            for s in S}

# Objective
master.setObjective(quicksum(g_s * l[s] for s in S), GRB.MAXIMIZE)

# Constraints
MaxPeople = {p: master.addConstr(quicksum(a[p, s] * lambda_s for s in S) <= n[p])
             for p in P}
ResourceCapacity = {h: master.addConstr(quicksum(b[h,s] * l[s]) <= c[h])
                    for h in H}
MaxAmbulances = master.addConstr(quicksum(l[s] for s in S) <= n_a)

