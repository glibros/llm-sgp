import json
import sympy
from timeout_decorator import timeout, TimeoutError
from json.decoder import JSONDecodeError

import deap.tools.llmparser as llmparser

# def initRepeat(container, func, n):
#     """Call the function *func* *n* times and return the results in a
#     container type `container`

#     :param container: The type to put in the data from func.
#     :param func: The function that will be called n times to fill the
#                  container.
#     :param n: The number of times to repeat func.
#     :returns: An instance of the container filled with data from func.

#     This helper function can be used in conjunction with a Toolbox
#     to register a generator of filled containers, as individuals or
#     population.

#         >>> import random
#         >>> random.seed(42)
#         >>> initRepeat(list, random.random, 2) # doctest: +ELLIPSIS,
#         ...                                    # doctest: +NORMALIZE_WHITESPACE
#         [0.6394..., 0.0250...]

#     See the :ref:`list-of-floats` and :ref:`population` tutorials for more examples.
#     """
#     return container(func() for _ in range(n))

sympyLocals = {
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
    # 'abs': lambda x: sympy.Abs(x),
    'asinh': lambda x: sympy.asinh(x),
    'acosh': lambda x: sympy.acosh(x),
    'atanh': lambda x: sympy.atanh(x),
}

@timeout(10)  # Set the timeout limit to 10 seconds
def safeSympify(expression):
    # If x not present, convert all string ints to floats
    result = ""
    current_number = ""
    if 'x' not in expression:
        for char in expression:
            if char.isdigit() or char == '.':
                current_number += char
            else:
                if current_number:
                    result += str(float(current_number))
                    current_number = ""
                result += char
        if current_number:
            result += str(float(current_number))
        return sympy.sympify(result, locals=sympyLocals)
    else:
        return sympy.sympify(expression, locals=sympyLocals)

@timeout(10)  
def safeSymplify(expression):
    return sympy.simplify(expression)

def wizardLLMInitialPopulation(llm, m):
    """ Get the initial population from llama according to well known problems and by using
    llm priors.
    """
            
    # Format arguments properly
    arguments = ""
    for arg in range(llm.dimensions):
        arguments += "x" + str(arg+1) + ", "
    # Remove extra comma
    arguments = arguments[:-2]

    message = ("The followng is the RMSE results of using different "
            "regression models to fit the data: "
            + llm.regressions + ". "
            "For a given Genetic Programming algorithm for a " +
            str(llm.dimensions) + "-dimensional (with " + arguments + " as argument(s)) "
            "Symbolic Regression problem, "
            "these regressions are a good indicatif of the type of function we are looking for. "
            "A lower RMSE closer to 0 means that the symbolic expression we are "
            "looking for could be in a similar form to the regression model. "
            "Based on the combination of these results, and according to well-known "
            "Symbolic Regression benchmarks (e.g., Koza, Keijzer, RatPol3D, "
            "Nguyen etc.), or real-world problems from physics, biology or chemistry, "
            "give examples of "+ str(m) + " extremely diverse initial genetic "
            "programming individuals in the form of mathematical expressions. "
            "Your final answer must be in the form of a single JSON object, with the key "
            "being the individual number (from '1' to '" + str(m) + "') "
            "and the value being the mathematical expression. Your expressions don't need "
            "to be perfectly accurate.")
    
    # LLM parameter to adjust in case of missing JSON object
    temperature = 0.2
    for i in range(10):
        try:
            llm.individuals = llmparser.find_json_in_text(llm.complete(message, temperature=temperature))
            if llm.individuals == None:
                print("initial llm individuals JSON cannot be None, retrying...")
                temperature += 0.01
                continue
            break
        except JSONDecodeError as e:
            print("Error decoding initiali JSON object: ", e)
            print("Retrying...")
            temperature += 0.01
    if llm.individuals == None or llm.individuals == "" or llm.individuals == r"{}":
        raise Exception("Could not find any valid JSON")

def llmGetInitialPopulation(llm, m):
        """ Get the initial population from the llm according to well known problems and by using
        llm priors.
        """

        # Reset the chat context to get clean population from llm
        llm.reset()

        # Set new context and give some information to get priors
        user = ("The followng is the RMSE (the lower the better) results of using different "
                "regression models to fit the data: " + llm.regressions + " "
                "These regressions are a strong indicatif of the type of function we are "
                "looking for.")
        assistant = ("A lower RMSE means the expression we are looking for is in a similar form "
                        "to the regression model.")
        llm.initialAnalysis(user, assistant)

        message = ("Estimate " + str(m) + " "
                "genetic programming individuals. Individuals must be initialized "
                "only according to well-known Symbolic Regression benchmarks (e.g., Koza, Keijzer, RatPol3D, "
                "Nguyen etc.), or real-world problems from physics, biology, chemistry, engineering. "
                "(e.g., Newton's law, Ohm's law, heat transfer, breeder's equation, etc.) "
                "Your answer must be in JSON, with the key being the individual index (from '1' to '"
                + str(m) + "') "
                "and the value being the mathematical expression. ")
                #    "{'1': 'expression1', '2': ..., '" + str(m) + "': 'expression" + str(m) + "'}.")
        llm.individuals = llm.addQuestion(message).content

def initRepeat(llm, container, func, func2, n, m):
    """Call the function *func* *m* (where m is the number of individuals
    generated by the llm) times and return the results in a container 
    type `container`. The function *func2* is called *n-m* times and the results
    are also returned in a container type `container`. The two containers are
    then concatenated and returned.

    :param container: The type to put in the data from func.
    :param func: The function that will be called n times to fill the
                container.
    :param n: The number of times to repeat func.
    :returns: An instance of the container filled with data from func.
    """
    # Fiirst get the initial population from llm (this will also set the pset)
    try:
        wizardLLMInitialPopulation(llm, m)
    except Exception as e:
        print("Fatal error. No individuals were created by the llm, exiting")
        exit(1)
        n = n + m
        m = 0
    # llmGetInitialPopulation(llm, m)
    # For debugging
    print("LLM Individuals: \n", llm.individuals)
    llmIndividuals = llm.individuals
    llmContainer = container()
    for i in range(m):
        try:
            inputExpression = llmIndividuals[str(i+1)]
            # inputExpression = inputExpression.replace('^', '**')
            ind = func(string=inputExpression)
            llmContainer.append(ind)
            llm.libraryAdd(ind, inputExpression)
        # If the llm individual is not valid, then this individual will be generated
        # With function func2
        except ValueError as e:
            if 'inputExpression' in locals():
                print("A value error occured in initRepeat parsing the expression: ", inputExpression)
                print("Error: ", e)
                print("Skipping it and using standard initialization instead.")
                del llmIndividuals[str(i+1)]
                n += 1
                continue
        except TypeError as e:
            if 'inputExpression' in locals():
                print("A value error occured in initRepeat parsing the expression: ", inputExpression)
                print("Error: ", e)
                print("Skipping it and using standard initialization instead.")
                del llmIndividuals[str(i+1)]
                n += 1
                continue
        except Exception as e:
            if 'inputExpression' in locals():
                print("A value error occured in initRepeat parsing the expression: ", inputExpression)
                print("Error: ", e)
                print("Skipping it and using standard initialization instead.")
                del llmIndividuals[str(i+1)]
                n += 1
                continue
    if m != 0:
        # Update llm individuals (since some of them might have been deleted)
        llm.individuals = json.dumps(llmIndividuals)
        
    # Then get the probabilities of choosing a primitive function (according to the selected primitives)
    # llm.context.reset()
    # llm.probabilities = self.llmUpdateProbabilities()
    # Then initialize the rest of the population using the biased initialization
    biasedContainer = container()
    for i in range(n-m):
        ind = func2()
        if ind.height < 4:
            try:
                st = str(ind)
                # sympyExpr = sympy.sympify(st, locals=sympyLocals)
                sympyExpr = safeSympify(st)
                sympyExpr = str(sympyExpr)
                llm.libraryAdd(ind, sympyExpr)
            except:
                pass
        biasedContainer.append(ind)

    # Return the concatenated containers
    return llmContainer + biasedContainer

    # return container(func(i) if i < m else func2() for i in range(n))


def initIterate(container, generator):
    """Call the function *container* with an iterable as
    its only argument. The iterable must be returned by
    the method or the object *generator*.

    :param container: The type to put in the data from func.
    :param generator: A function returning an iterable (list, tuple, ...),
                      the content of this iterable will fill the container.
    :returns: An instance of the container filled with data from the
              generator.

    This helper function can be used in conjunction with a Toolbox
    to register a generator of filled containers, as individuals or
    population.

        >>> import random
        >>> from functools import partial
        >>> random.seed(42)
        >>> gen_idx = partial(random.sample, range(10), 10)
        >>> initIterate(list, gen_idx)      # doctest: +SKIP
        [1, 0, 4, 9, 6, 5, 8, 2, 3, 7]

    See the :ref:`permutation` and :ref:`arithmetic-expr` tutorials for
    more examples.
    """
    return container(generator())


def initCycle(container, seq_func, n=1):
    """Call the function *container* with a generator function corresponding
    to the calling *n* times the functions present in *seq_func*.

    :param container: The type to put in the data from func.
    :param seq_func: A list of function objects to be called in order to
                     fill the container.
    :param n: Number of times to iterate through the list of functions.
    :returns: An instance of the container filled with data from the
              returned by the functions.

    This helper function can be used in conjunction with a Toolbox
    to register a generator of filled containers, as individuals or
    population.

        >>> func_seq = [lambda:1 , lambda:'a', lambda:3]
        >>> initCycle(list, func_seq, n=2)
        [1, 'a', 3, 1, 'a', 3]

    See the :ref:`funky` tutorial for an example.
    """
    return container(func() for _ in range(n) for func in seq_func)


__all__ = ['initRepeat', 'initIterate', 'initCycle', 
           'llmGetInitialPopulation', 'sympyLocals', 'safeSympify',
           'safeSymplify', 'wizardLLMInitialPopulation']


if __name__ == "__main__":
    import doctest
    import random
    random.seed(64)
    doctest.run_docstring_examples(initRepeat, globals())

    random.seed(64)
    doctest.run_docstring_examples(initIterate, globals())
    doctest.run_docstring_examples(initCycle, globals())
