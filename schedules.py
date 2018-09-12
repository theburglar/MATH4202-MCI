from gurobipy import *
import random
from pprint import pprint

I = 0
D = 1

# random.seed(0)
# #number of immediate patients
# n_i = random.randint(10, 30)
# #number of delayed patients
# n_d = random.randint(10, 30)
# #number of ambulances
# n_a = random.randint(10, 30)
# #resources required for immediate patients
# w_i = random.randint(3, 6)
# #resources required for delayed patients
# w_d = random.randint(1, 3)
# required_resources = n_i * w_i + n_d * w_d
#
# #resources available per hospital
# c_h = []
# while sum(c_h) < required_resources:
#     # c_h.append(random.randint(10, 30))



###############################################
n_i = 3
n_d = 4
n_a = 5
w_i = 3
w_d = 3
c_h = [10 for i in range(5)]

###############################################

#time to hospital
t_h = [random.randint(5,50) for h in c_h]

H = range(len(c_h))

def generate_schedules(schedule, resources_h, p_i, p_d):
    schedules = []
    more = False
    for h in H:
        if resources_h[h] >= w_i and p_i > 0:
            new_schedule = schedule[:] + [(I, h)]
            new_resources = resources_h[:]
            new_resources[h] -= w_i
            schedules.extend(generate_schedules(new_schedule, new_resources, p_i-1, p_d))
            more = True
        if resources_h[h] >= w_d and p_d > 0:
            new_schedule = schedule[:] + [(D, h)]
            new_resources = resources_h[:]
            new_resources[h] -= w_d
            schedules.extend(generate_schedules(new_schedule, new_resources, p_i, p_d-1))
            more = True
    if not more:
        return [schedule]
    return schedules

def analyse_schedule(schedule):
    g = 0
    a = [0, 0]
    b_h = [0 for h in H]

    t = 0
    for p, h in schedule:
        t += t_h[h]
        g += p_survival(t)
        a[p] += 1
        b_h[h] += w_i if p == I else w_d
        t += t_h[h]
    return (g, a, b_h)


ss = generate_schedules([], c_h, n_i, n_d)
for s in ss:
    print(s)
print(len(ss))

# w = {I: w_i, D: w_d}
# n = {I: n_i, D: n_d}
#
# def bitchin_it():
#     schedules = []
#     resources_h = c_h[:]
#     p_i = n_i
#     p_d = n_d
#
#     new_schedules = []
#     for schedule in schedules:
#         for h in H:
#             for p in (I, D):
#                 if resources_h[h] >= w[p] and n[p] > 0:
#
#                     schedules.append()
#                     resources_h[h] -= w_i



