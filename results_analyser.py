from pprint import pprint
from statistics import stdev
import numpy as np

import matplotlib.pyplot as plt

TESTS = ('combined_closest',
         'combined_first',
         'delayed_closest',
         'delayed_first',
         'immediate_closest',
         'immediate_first',
         'none_closest',
         'none_first')

SCENARIO_LABELS = {'MOD': 'Moderate',
                   'OPT': 'Optimistic',
                   'PES': 'Pessimistic'}

def get_fnp_data():
    data = {}
    for test_name in TESTS:
        data[test_name] = {}
        for filename in [f'{scen}_{i}' for scen in ('OPT', 'MOD', 'PES') for i in range(0, 100)]:
            with open(f'test_results/fix_and_price/{test_name}/{filename}_results.txt') as file:
                res = {}
                for line in file:
                    if line.startswith('Optimal Value Determined'):
                        res['obj'] = float(line.split(' ')[-1])
                    if line.startswith('Linear objective'):
                        res['linear'] = float(line.split(' ')[-1].partition('%')[0])
                    if line.startswith('Total time taken'):
                        res['time'] = float(line.split(' ')[-2])
                res['gap'] = (res['linear'] - res['obj']) / res['obj'] * 100
                data[test_name][filename] = res
    return data


def get_best_fix_and_price_obj(data):
    res = {}
    for case in [f'{scen}_{i}' for scen in ('OPT', 'MOD', 'PES') for i in range(0, 100)]:
        res[case] = max((data[test][case]['obj'], test) for test in TESTS)
    return res


def get_heuristic_data():
    data = {}
    for test_name in ('delayed', 'immediate'):
        data[test_name] = {}
        for filename in [f'{scen}_{i}' for scen in ('OPT', 'MOD', 'PES') for i in range(0, 100)]:
            with open(f'test_results/heuristic_rip/{test_name}/{filename}_results.txt') as file:
                res = {}
                for line in file:
                    if line.startswith('Optimal Value Determined'):
                        res['obj'] = float(line.split(' ')[-1])
                    if line.startswith('Total time taken'):
                        res['time'] = float(line.split(' ')[-2])
                data[test_name][filename] = res
    return data


def get_heuristic_gap(h_data, fnp_obj_data):
    res = {}
    for i in ('delayed', 'immediate'):
        res[i] = {}
        for k in h_data[i]:
            res[i][k] = (fnp_obj_data[k][0] - h_data[i][k]['obj']) / fnp_obj_data[k][0] * 100
    return res


def get_heuristic_gap_graph_data(gap_data):
    res = {'delayed': {},
           'immediate': {}}
    for filename in [f'{scen}_{i}' for scen in ('OPT', 'MOD', 'PES') for i in range(0, 100)]:
        with open(f'test_cases/{filename}.txt') as file:
            n_p = float(file.readline().strip())
            n_p += float(file.readline().strip())
            n_a = float(file.readline().strip())
            file.readline(); file.readline(); file.readline()
            times = [int(t) for t in file.readline().strip().split(',')]
            average_time = sum(times) / len(times)
        for i in ('delayed', 'immediate'):
            res[i][filename] = ((n_p * average_time / n_a), gap_data[i][filename])

    return res


def full_heuristic_gap_graph_data():
    x = get_fnp_data()
    y = get_heuristic_data()
    z = get_best_fix_and_price_obj(x)
    q = get_heuristic_gap(y, z)
    data = get_heuristic_gap_graph_data(q)

    res = {}
    for i in ('immediate', 'delayed'):
        res[i] = {'MOD': {},
                  'OPT': {},
                  'PES': {}}
        for k in data[i]:
            for scen in ('MOD', 'OPT', 'PES'):
                if k.startswith(scen):
                    res[i][scen][k] = data[i][k]
    return res


def heuristic_scatter_plot(p, scen):
    data = full_heuristic_gap_graph_data()[p][scen]
    x = [data[k][0] for k in data]
    y = [data[k][1] for k in data]
    plt.scatter(x,y)
    plt.ylim(0, 100)
    plt.title(f'{p.capitalize()}-first: {SCENARIO_LABELS[scen]}')
    plt.xlabel('Average Schedule Time per Ambulance')
    plt.ylabel('Gap %')
    # plt.show()
    plt.savefig(f'C:\\Users\\Rudi\\Documents\\Uni\\MATH4202\\Figures\\scatter_{p}_firsrt_{scen}.jpeg')

def get_fnp_table_stats(fnp_data):
    pass


def get_alpha_comparison_data(fnp_data):
    res = {}
    for generation in ('combined', 'immediate', 'delayed', 'none'):
        res[generation] = {}
        for alpha in ('first', 'closest'):
            res[generation][alpha] = {}
            for scen in ('MOD', 'OPT', 'PES'):
                objs = []
                times = []
                for i in range(100):
                    objs.append(fnp_data[f'{generation}_{alpha}'][f'{scen}_{i}']['obj'])
                    times.append(fnp_data[f'{generation}_{alpha}'][f'{scen}_{i}']['time'])
                avg_obj = sum(objs) / len(objs)
                avg_time = sum(times) / len(times)
                max_time = max(times)
                std_time = stdev(times)
                res[generation][alpha][scen] = {'avg_obj': avg_obj,
                                          'avg_time': avg_time,
                                          'max_time': max_time,
                                          'std_time': std_time}
    return res


def bar_chart_alpha_comparison_obj(generation):
    fnp_data = get_fnp_data()
    data = get_alpha_comparison_data(fnp_data)[generation]

    first_times = [data['first'][k]['avg_obj'] for k in data['first']]
    closest_times = [data['closest'][k]['avg_obj'] for k in data['closest']]

    bar_width = 0.35
    index = np.arange(3)

    plt.figure()

    plt.bar(index, first_times, bar_width, color='b', label='First')
    plt.bar(index + bar_width, closest_times, bar_width, color='g', label='Closest')
    plt.xticks(index + bar_width/2, ('Moderate', 'Optimistic', 'Pessimistic'))
    plt.xlabel('Scenarios')
    plt.ylabel('Average Objective Value')

    plt.title(f'Different Threshold Vectors: {generation.capitalize()} Generation')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'C:\\Users\\Rudi\\Documents\\Uni\\MATH4202\\Figures\\bar_alphas_{generation}_obj.jpeg')


def bar_chart_alpha_comparison_time(generation):
    fnp_data = get_fnp_data()
    data = get_alpha_comparison_data(fnp_data)[generation]

    first_times = [data['first'][k]['avg_time'] for k in data['first']]
    closest_times = [data['closest'][k]['avg_time'] for k in data['closest']]

    bar_width = 0.35
    index = np.arange(3)

    plt.figure()

    plt.bar(index, first_times, bar_width, color='b', label='First')
    plt.bar(index + bar_width, closest_times, bar_width, color='g', label='Closest')
    plt.xticks(index + bar_width/2, ('Moderate', 'Optimistic', 'Pessimistic'))
    plt.xlabel('Scenarios')
    plt.ylabel('Average Time Taken (seconds)')

    plt.title(f'Different Threshold Vectors: {generation.capitalize()} Generation')
    plt.legend()
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'C:\\Users\\Rudi\\Documents\\Uni\\MATH4202\\Figures\\bar_alphas_{generation}_time.jpeg')


def get_schedule_comparison_data(fnp_data):
    res = {}
    for alpha in ('first', 'closest'):
        res[alpha] = {}
        for generation in ('combined', 'immediate', 'delayed', 'none'):
            res[alpha][generation] = {}
            for scen in ('MOD', 'OPT', 'PES'):
                objs = []
                times = []
                for i in range(100):
                    objs.append(fnp_data[f'{generation}_{alpha}'][f'{scen}_{i}']['obj'])
                    times.append(fnp_data[f'{generation}_{alpha}'][f'{scen}_{i}']['time'])
                avg_obj = sum(objs) / len(objs)
                avg_time = sum(times) / len(times)
                max_time = max(times)
                std_time = stdev(times)
                res[alpha][generation][scen] = {'avg_obj': avg_obj,
                                                'avg_time': avg_time,
                                                'max_time': max_time,
                                                'std_time': std_time}
    return res

# fnp_data = get_fnp_data()
# h_data = get_heuristic_data()
# a_data = get_alpha_comparison_data(fnp_data)
# s_data = get_schedule_comparison_data(fnp_data)
# pprint(s_data)

def all_scatter_plots():
    for i in ('immediate', 'delayed'):
        for scen in ('OPT', 'MOD', 'PES'):
            heuristic_scatter_plot(i, scen)

def all_alpha_comparison_bar_charts():
    for generation in ('combined', 'delayed', 'immediate', 'none'):
        bar_chart_alpha_comparison_obj(generation)
    for generation in ('combined', 'delayed', 'immediate', 'none'):
        bar_chart_alpha_comparison_time(generation)

all_alpha_comparison_bar_charts()