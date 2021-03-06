# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 20:03:03 2018

@author: rosie
"""
from random import randint
import os.path

NUM_CASES = 100
SCENARIOS = ['PES','MOD','OPT']


tests= []
for i in range(NUM_CASES):
    n_i = randint(10,30)
    n_d = randint(10,30)
    n_a = randint(10,30)
    w_i = randint(3,6)
    w_d = randint(1,3)
    c = []
    required_resources = n_i * w_i + n_d * w_d
    while sum(c) < required_resources:
        c.append(randint(10,30))
    times = [str(randint(5,50)) for h in c]
    c_input = ",".join([str(x) for x in c])
    times_input = ",".join(times)
    inputs = [str(x) for x in [n_i,n_d,n_a,w_i,w_d]]
    inputs.append(c_input)
    inputs.append(times_input)
    tests = inputs[:]

   # print(inputs)
   
    save_path = "test_cases"
    for SCENARIO in range(3):
        file_name = SCENARIOS[SCENARIO]+"_"+str(i)
        completeName = os.path.join(save_path, file_name+".txt")
        file = open(completeName, "w")
        
        
        testCopy = tests[:]
        testCopy.append(str(SCENARIO))
        
        content = "\n".join(testCopy)
        
        
        file.write(content)
        file.close()