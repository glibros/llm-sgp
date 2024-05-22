import random
import warnings
import operator
import time
import os
import sympy
import math
import argparse
from LoadDataset import functions

# from llm import LLM
from InitialAnalysis import InitialAnalysis

from deap import base, creator, tools, gp, algorithms, llm
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import multiprocessing

def protected_div(left, right):
    if right == 0.0:
        return left
    res = left / right
    if res > 1e7:
        return 1e7
    if res < -1e7:
        return -1e7
    return res

def protected_mul(left, right):
    try:
        return left*right
    except OverflowError:
        return 1e7

def protected_sqrt(x):
    if x < 0.0:
        return 0.0
    return math.sqrt(x)

def protected_log(x):
    if abs(arg) < 1e-5:
        arg = 1e-5
    return math.log(abs(arg))

def exp(x):
    if x > 100:
        return 100
    else:
        return math.exp(x)    


def randomConstant():
    return random.randint(0, 10)

possible_primitives = {
    'add': [operator.add, 2],
    'sub': [operator.sub, 2],
    'mul': [operator.mul, 2],
    'protected_div': [protected_div, 2],
    'protected_sqrt': [protected_sqrt, 1],
    'protected_log': [protected_log, 1],
    'exp': [exp, 1],
    'sin': [math.sin, 1],
    'cos': [math.cos, 1],
    'tan': [math.tan, 1],
    'sinh': [math.sinh, 1],
    'cosh': [math.cosh, 1],
    'tanh': [math.tanh, 1],
    'asin': [math.asin, 1],
    'acos': [math.acos, 1],
    'atan': [math.atan, 1],
    # 'abs': [operator.abs, 1],
    'asinh': [math.asinh, 1],
    'acosh': [math.acosh, 1],
    'atanh': [math.atanh, 1]
}

famous_constants = {
    'pi': math.pi,
    'e': math.e,
    'E': math.e,
    'phi': 1.618033988749895,
    'gamma': 0.5772156649015329,
    'c': 299792458,
    'G': 6.67430e-11,
    'h': 6.62607015e-34,
    'hbar': 1.054571817e-34,
    'k': 1.380649e-23,
    'sigma': 5.670374419e-8,
    'R': 8.314462618,
    'alpha': 0.0072973525693,
    'Na': 6.02214076e23,
    'mu0': 1.25663706212e-6,
    'epsilon0': 8.8541878128e-12,
    # 'I': 1j,
}

locs = {
    'sub': lambda x, y: x - y,
    'protected_div': lambda x, y: x/y,
    'mul': lambda x, y: x*y,
    'add': lambda x, y: x + y,
    'sin': lambda x: sympy.sin(x),
    'cos': lambda x: sympy.cos(x),
    'tan': lambda x: sympy.tan(x),
    'protected_sqrt': lambda x: sympy.sqrt(x),
    'protected_log': lambda x: sympy.ln(x),
    'exp': lambda x: sympy.exp(x),
    'sinh': lambda x: sympy.sinh(x),
    'cosh': lambda x: sympy.cosh(x),
    'tanh': lambda x: sympy.tanh(x),
    'asin': lambda x: sympy.asin(x),
    'acos': lambda x: sympy.acos(x),
    'atan': lambda x: sympy.atan(x),
    'abs': lambda x: sympy.Abs(x),
    'asinh': lambda x: sympy.asinh(x),
    'acosh': lambda x: sympy.acosh(x),
    'atanh': lambda x: sympy.atanh(x),
}

# pool = multiprocessing.Pool()s

# Create the fitness and individual classes (for RMSE)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# Create the toolbox
toolbox = base.Toolbox()

# For multi threading
# toolbox.register("map", pool.map)

# Define the evaluation function
def evaluate(individual, x_features, data, pset):
    func = gp.compile(expr=individual, pset=pset)

    # Predict using the symbolic expression
    y_pred = []
    for x in x_features:
        try:
            y_pred.append(func(*x))
        except:
            y_pred.append(500)

    # Calculate root mean squared error
    try:
        rmse = np.sqrt(mean_squared_error(data["y"], y_pred))
    except:
        rmse = 500

    # Return the root mean squared error as the fitness
    return rmse,


def main():
    # When running final tests, run it through a shelll for changing the seed,
    # the problem number
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=int, default=1, help="Problem number to run")
    parser.add_argument("--run", type=int, default=1, help="Which run (also used as seed)")
    
    args = parser.parse_args()
    problem = args.problem
    run = args.run
    
    # Create results list 
    # try:
    # For which problem to run (set in the bash script)
    # for n in range(6, 7):
    
    # File to save the results to
    resultsFile = "./results/problem_" + str(problem) + ".txt"
    # Check if the file exists
    fileExists = os.path.isfile(resultsFile)
    
    # Individual run (seed)
    # for run in range(0, 1):
    
    results = []
    seed = 0 + run
    random.seed(0 + run, version=2)
    # Load data
    train, test = functions[problem](seed)
    dimensions = train.shape[1] - 1

    # Ignore warnings for the initial analysis
    warnings.filterwarnings('ignore')
    # Prior analysis
    analysis = InitialAnalysis(train)
    # Reset warnings
    warnings.resetwarnings()

    def randomConstant():
        return random.uniform(-50.0, 50.0)

    # The primitive set to use should be set according to the llm
    # Warning: the order of the primitives should be the same as the order of the probabilities
    pset = gp.PrimitiveSet("MAIN", dimensions)
    # pset.addEphemeralConstant("rand101", randomConstant)

    # Rename arguments
    for i in range(dimensions):
        pset.renameArguments(**{f'ARG{i}': f'x{i+1}'})
    
    toolbox.register("compile", gp.compile, pset=pset)

    llmObj = llm.LLM(dimensions=train.shape[1] - 1, regressions=str(analysis.models), seed=seed,
                        toolbox=toolbox, points=train.drop("y", axis=1).values.tolist())

    # Use custom llmGP class
    # llmGP = LLMGP(pset, train.shape[1] - 1, regressions=str(analysis.models),
    #               corr=str(analysis.corr), seed=42)

    # Define the individual and population
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("individual", tools.initIterate,
                    creator.Individual, toolbox.expr)
    toolbox.register("fromLLM", gp.fromSympyStr, container=creator.Individual, 
                    pset=pset, possiblePset=possible_primitives,
                    constants=famous_constants)
    toolbox.register("population", tools.initRepeat, llm=llmObj,
                    container=list, func=toolbox.fromLLM, 
                    func2=toolbox.individual)

    # Get the x features from the data DataFrame
    x_features = train.drop("y", axis=1).values

    # Update the evaluation function registration
    toolbox.register("evaluate", evaluate, x_features=x_features, data=train, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Custom new operators
    toolbox.register("llamaUpdate", gp.llamaUpdate, func=toolbox.fromLLM,
                    llm=llmObj)
    toolbox.register("llamaSimplify", gp.llamaSimplify, llm=llmObj, 
                        func=toolbox.fromLLM)
    toolbox.register("llmRDO", gp.llmRandomDesiredOperator, llm=llmObj,
                        tarSem=train["y"].to_numpy(), 
                        points=train.drop("y", axis=1).values.tolist(),
                        toolbox=toolbox)

    toolbox.decorate("mate", gp.staticLimit(
        key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(
        key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("llmRDO",gp.staticLimit(
        key=operator.attrgetter("height"), max_value=17) )

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("med", np.median)
    stats.register("min", np.min)
    stats.register("max", np.max)

    suc = 0
    evaTimes = 0
    trainErr = 1e6
    testErr = 1e6
    progSize = 0
    trainTime = 0

    start = time.time()

    # Initialize a population and evolve
    pop = toolbox.population(n=50, m=15)
    hof = tools.HallOfFame(1)

    # Evolve the population
    pop, log = algorithms.llmEa(pop, toolbox, cxpb=0.5, mutpb=0.1, rdopb=0.5, llmspb=0.5, llmupb=0.6, 
                                llmIndS=6, ngen=100, stats=stats, halloffame=hof, verbose=True)

    end = time.time()

    conv = [log.select("nevals"), log.select("NNE"), log.select("min"),
            log.select("avg"), log.select("med"),
            log.select("simplified"), log.select("replaced"), log.select("optimized")]
    evaTimes = sum(log.select("nevals"))
    nne = sum(log.select("NNE"))
    # simplified = log.select("simplified")
    # replaced = log.select("replaced")
    # optimized = log.select("optimized")

    if hof.items[0].fitness.values[0] < 1e-4:
        suc = 1
    trainErr = hof.items[0].fitness.values[0]
    progSize = hof.items[0].__len__()
    trainTime = end - start

    # Test the best individual on unseen data
    testErr = evaluate(hof.items[0], test.drop("y", axis=1).values, test, pset=pset)[0]

    # Return the best individual
    best_expr = hof[0]
    best_expr_str = str(gp.PrimitiveTree(best_expr))

    sympy_expr = sympy.sympify(best_expr_str, locals=locs)
    try:
        sympy_expr_simp = tools.safeSymplify(sympy_expr)
    except:
        sympy_expr_simp = sympy_expr

    # Dictionary holding the results
    logDict = {
        "run": run,
        "suc": suc,
        "evals": evaTimes,
        "NNE": nne,
        # "individuals simplified": simplified,
        # "individuals replaced": replaced,
        # "RDO optimized": optimized,
        "train err": trainErr,
        "test err": testErr,
        "program length": progSize,
        "train time": trainTime,
        "best solution": sympy_expr,
        "simplified best": sympy_expr_simp
    }

    # Append this run's results
    results.append(logDict)

    # Convert to pandas dataframe
    df = pd.DataFrame(results)

    # Write the results to file
    if fileExists:
        df.to_csv(resultsFile, mode='a', index=False, header=False, sep='\t')
    else:
        df.to_csv(resultsFile, mode='w', index=False, header=True, sep='\t')

    # Dict holding the convergence data
    convDict = {
        "nevals": conv[0],
        "NNE": conv[1],
        "min": conv[2],
        "avg": conv[3],
        "med": conv[4],
        "individuals simplified": conv[5],
        "individuals replaced": conv[6],
        "RDO optimized": conv[7]
    }

    # Convert to pandas dataframe
    df = pd.DataFrame(convDict)

    # Write the results to file
    convFile = "./results/problem_" + str(problem) + "_conv.txt"
    fileExists = os.path.isfile(convFile)
    if fileExists:
        df.to_csv(convFile, mode='a', index=False, header=False, sep='\t')
    else:
        df.to_csv(convFile, mode='w', index=False, header=True, sep='\t')
            
# except Exception as e:
    #     # Log the exception
    #     print(f"An exception has occured: {e}")
        
    #     # Convert to pandas dataframe and write to file
    #     resultsFile = "./results/problem_" + str(n) + ".csv"
    #     fileExists = os.path.isfile(resultsFile)
    #     df = pd.DataFrame(results)

    #     # Write the results to file
    #     if fileExists:
    #         df.to_csv(resultsFile, mode='a', index=False, header=False)
    #     else:
    #         df.to_csv(resultsFile, mode='w', index=False, header=True)

if __name__ == "__main__":
    main()
