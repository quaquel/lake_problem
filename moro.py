'''


'''
from __future__ import (unicode_literals, print_function, division)

import math
import numpy as np
from scipy.optimize import brentq as root

from ema_workbench.em_framework import (RealParameter, ScalarOutcome, Model,
                                        Constant)
from ema_workbench.util import ema_logging

from borg_python import borg

from borg_ro_setup import RobustProblem

# Created on Jul 4, 2016
#
# .. codeauthor::jhkwakkel <j.h.kwakkel (at) tudelft (dot) nl>

def lake_problem(
         b = 0.42,          # decay rate for P in lake (0.42 = irreversible)
         q = 2.0,           # recycling exponent
         mean = 0.02,       # mean of natural inflows
         stdev = 0.001,     # future utility discount rate
         delta = 0.98,      # standard deviation of natural inflows
         alpha = 0.4,       # utility from pollution
         nsamples = 100,    # Monte Carlo sampling of natural inflows
         **kwargs):   
    decisions = [kwargs[str(i)] for i in range(100)]
    
    Pcrit = root(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)
    nvars = len(decisions)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(decisions)
    reliability = 0.0

    for _ in xrange(nsamples):
        X[0] = 0.0
        
        natural_inflows = np.random.lognormal(
                math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
                math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
                size = nvars)
        
        for t in xrange(1,nvars):
            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decisions[t-1] + natural_inflows[t-1]
            average_daily_P[t] += X[t]/float(nsamples)
    
        reliability += np.sum(X < Pcrit)/float(nsamples*nvars)
      
    max_P = np.max(average_daily_P)
    utility = np.sum(alpha*decisions*np.power(delta,np.arange(nvars)))
    inertia = np.sum(np.diff(decisions) > -0.02)/float(nvars-1)
    
    return max_P, utility, inertia, reliability



def calculate_robustness(outcomes):
    '''calculate robustness scores
    
    here we use the mean and the std as objectives
    
    ..note:: other metrics can be tested easily by extending this class
    and overriding this one method
    
    '''
    
    objs = []
    for entry in ["max_P", 'utility', 'inertia', 
                  'reliability']:
        
        minimize = -1
        if entry == 'max_P':
            minimize = 1
        
        outcome = outcomes[entry]
        mean_performance = minimize * np.mean(outcome)
        std_performance = np.std(outcome)
        objs.append(float(mean_performance))
        objs.append(float(std_performance))
    
    return objs

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    
    #instantiate the model
    model = Model('lakeproblem', function=lake_problem)
    
    #specify uncertainties
    model.uncertainties = [RealParameter("b", 0.1, 0.45),
                           RealParameter("q", 2.0, 4.5),
                           RealParameter("mean", 0.01, 0.05),
                           RealParameter("stdev", 0.001, 0.005),
                           RealParameter("delta", 0.93, 0.99)]
    #specify outcomes 
    model.outcomes = [ScalarOutcome("max_P",),
                      ScalarOutcome("utility"),
                      ScalarOutcome("inertia"),
                      ScalarOutcome("reliability")]
    
    # override some of the defaults of the model
    model.constants = [Constant('alpha', 0.41),
                       Constant('nsamples', 150),]
    
    # set levers, one for each time step
    model.levers = [RealParameter(str(i), 0, 0.05) for i in range(100)]
    
    problem = RobustProblem(model, calculate_robustness, parallel=True,
                            processes=None, nr_exp=100)

    nvars = problem.nvars    
    nobjs = problem.nobjs * 2
    borg_algorithm = borg.Borg(nvars, nobjs, 0, problem.obj_func)

    borg_algorithm.epsilons[:] = 0.05
    
    for i, entry in enumerate(problem.levers):
        borg_algorithm.bounds[i] = entry.params
        
    result = borg_algorithm.solve(maxEvaluations=100000,
                                  initialPopulationSize=25,
                                  minimumPopulationSize=25,
                                  frequency=5)
         
    res = result.to_dataframe()
    stats = result.statistics
     
    res.to_csv('./data/archive501.csv')
    stats.to_csv('./data/statistics501.csv')