from gurobipy import *
import time

EPSILON = 10**-5

I = 0
D = 1

PATIENT = {I: 'immediate',
           D: 'delayed'}

PES = 0
MOD = 1
OPT = 2

SURVIVAL_RATES = {
    PES: ((0.09, 17, 1.10), (0.57, 61, 2.03)),
    MOD: ((0.24, 47, 1.30), (0.76, 138, 2.17)),
    OPT: ((0.56, 91, 1.58), (0.81, 160, 2.41))
}

TEST_CASES = [f'{scen}_{i}' for scen in ('OPT','MOD','PES') for i in range(100)]

##################################################################
# Which Heuristic to run
HEURISTIC = D
##################################################################


# Helper Functions
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

# Schedule Generation
def generate_heuristic_schedules_immediate():

    n_d_remaining = n_d
    n_i_remaining = n_i

    resources_remaining = c[:]

    visit_order = []

    while n_i_remaining > 0:

        # get nearest available hospital
        hospital = None
        best_time = 100
        for i, r in enumerate(resources_remaining):
            if r >= w_i and times[i] < best_time:
                hospital = i
                best_time = times[i]

        if hospital is None:
            break
        # add to schedule
        visit_order.append(hospital)
        resources_remaining[hospital] -= w_i
        n_i_remaining -= 1

    cutoff = len(visit_order)

    while n_d_remaining > 0:

        # get nearest available hospital
        hospital = None
        best_time = 100
        for i, r in enumerate(resources_remaining):
            if r >= w_d and times[i] < best_time:
                hospital = i
                best_time = times[i]

        if hospital is None:
            break
        # add to schedule
        visit_order.append(hospital)
        resources_remaining[hospital] -= w_d
        n_d_remaining -= 1

    schedules = []
    for i in range(n_a):
        patient = I
        j = i
        if j >= cutoff:
            patient = D
        if j >= len(visit_order):
            break
        sched = [(patient, visit_order[j])]

        while True:
            j += n_a
            if j >= cutoff:
                patient = D
            if j >= len(visit_order):
                schedules.append(tuple(sched))
                break
            sched.append((patient, visit_order[j]))

    return schedules


def generate_heuristic_schedules_delayed():

    n_d_remaining = n_d
    n_i_remaining = n_i

    resources_remaining = c[:]

    visit_order = []

    while n_d_remaining > 0:

        # get nearest available hospital
        hospital = None
        best_time = 100
        for i, r in enumerate(resources_remaining):
            if r >= w_d and times[i] < best_time:
                hospital = i
                best_time = times[i]

        if hospital is None:
            break
        # add to schedule
        visit_order.append(hospital)
        resources_remaining[hospital] -= w_d
        n_d_remaining -= 1

    cutoff = len(visit_order)

    while n_i_remaining > 0:

        # get nearest available hospital
        hospital = None
        best_time = 100
        for i, r in enumerate(resources_remaining):
            if r >= w_i and times[i] < best_time:
                hospital = i
                best_time = times[i]

        if hospital is None:
            break
        # add to schedule
        visit_order.append(hospital)
        resources_remaining[hospital] -= w_i
        n_i_remaining -= 1

    schedules = []
    for i in range(n_a):
        patient = D
        j = i
        if j >= cutoff:
            patient = I
        if j >= len(visit_order):
            break
        sched = [(patient, visit_order[j])]

        while True:
            j += n_a
            if j >= cutoff:
                patient = I
            if j >= len(visit_order):
                schedules.append(tuple(sched))
                break
            sched.append((patient, visit_order[j]))

    return schedules

batch_start = time.time()
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

    start_time = time.time()

    generate_start = time.time()

    if HEURISTIC == I:
        schedules = generate_heuristic_schedules_immediate()
    else:
        schedules = generate_heuristic_schedules_delayed()

    print('Initial schedules generated...')
    generate_time = time.time() - generate_start

    master = Model('Master Problem')
    master.setParam('OutputFlag', 0)

    # Variables
    lambda_s = {s: master.addVar(vtype=GRB.INTEGER)
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

    master.optimize()

    duration = time.time() - start_time

    print(f'Successfully ran test case `{test_case}`')

    for s in lambda_s:
        if lambda_s[s].x > EPSILON:
            print(f'{round(lambda_s[s].x, 3)} lot(s) of', s)
    print('#' * 50)

    print('Optimal Value Determined:', master.objVal)
    print(f'Time generating schedules: {generate_time} seconds')
    print(f'Total time taken: {duration} seconds')
    print('#'*50)

    with open(f'test_results/heuristic_rip/{PATIENT[HEURISTIC]}/{test_case}_results.txt', 'w') as test_result:
        for s in lambda_s:
            if lambda_s[s].x > EPSILON:
                test_result.write(f'{round(lambda_s[s].x, 3)} lot(s) of {s}\n')
        test_result.write(f'Optimal Value Determined: {master.objVal}\n')
        test_result.write(f'Time generating schedules: {generate_time} seconds\n')
        test_result.write(f'Total time taken: {duration} seconds')

print('=' * 50)
print(f'Batch Time Taken: {time.time() - batch_start}')