import re
import json
from latex2sympy2 import latex2sympy, latex2latex
from timeout_decorator import timeout, TimeoutError

@timeout(10)  # Set the timeout limit to 10 seconds
def safeLatex2Sympy(expression):
    latex2sympy(expression)

def find_math_expressions(text):
    # Define a pattern to match expressions after "is:", "get:", "becomes:", "obtain:"
    # "=:". 
    # This particular pattern assumes that the expression will end with a 
    # sentence-ending punctuation or a newline. Adjust the pattern as needed.
    # pattern = r'(?:is|get|becomes|obtain):\s*([^\.,;\n]+(?:\.\d+)?)'
    # pattern = r'(?:is|get|becomes|obtain):\s*([^.,;\n]+(?:\.\d+)?)|=\s*([^.,;\n]+(?:\.\d+)?)'
    # pattern = r'(?:is|get|becomes|obtain):\s*([^.,;\n]+(?:\.\d+(?:[^.,;\n]+)?)?)|=\s*([^.,;\n]+(?:\.\d+(?:[^.,;\n]+)?)?)'
    pattern = r'(?:is|get|becomes|obtain):\s*([^.,;\n]+(?:\.\d+(?:[^.,;\n]+)?)*)|=\s*([^.,;\n]+(?:\.\d+(?:[^.,;\n]+)?)*)'

    # Find all instances
    matches = re.findall(pattern, text, re.MULTILINE)

    # Each match is a tuple, so we iterate through and pick the non-empty match
    expressions = []
    for match in matches:
        for expr in match:
            if expr:  # Filter out empty strings
                expressions.append(expr.strip())

    # Get the last expresdsion with incurrence of 'x'
    for expression in expressions[::-1]:
        if 'x' in expression:
            solution = expression
            break

    if 'solution' not in locals():
        raise Exception("No suitable math expression found when parsing text")

    # In case the LLM gave a latex expression
    if '\\' in solution:
        # Convert x indeces to latex compatible syntax, e.g., x1 -> x_1, x2 -> x_2, etc.
        indeces = []
        for index, char in enumerate(solution):
            if char == 'x':
                if solution[index + 1].isdigit():
                    indeces.append(index + 1)
        for index in indeces[::-1]:
            solution = solution[:index] + '_' + solution[index:]
                    
        solution = safeLatex2Sympy(solution)
        # blA = str(blA).replace('_', '') # This 'delimiter' is needed
        # print(blA)
    else:
        parts = solution.split('=')
        if len(parts) >= 2:
            solution = parts[-1].strip()
        parts = solution.split('$')
        if len(parts) >= 2:
            for part in parts[::-1]:
                if 'x' in part:
                    solution = part.strip()
                    break
        # Convert x indeces to latex compatible syntax, e.g., x1 -> x_1, x2 -> x_2, etc.
        indeces = []
        for index, char in enumerate(solution):
            if char == 'x':
                # Bound checking
                if index + 1 < len(solution):
                    if solution[index + 1].isdigit():
                        indeces.append(index + 1)
        for index in indeces[::-1]:
            solution = solution[:index] + '_' + solution[index:]
        if ('x' in solution) and len(indeces) == 0 and ("x_1" not in solution) and ("exp" not in solution):
            indeces = [i for i, ltr in enumerate(solution) if ltr == 'x']
            for index in indeces[::-1]:
                solution = solution[:index+1] + '_1' + solution[index+1:]
        # print(solution)

    return solution


def find_json_in_text(text):
        # For debugging 
        # print("Extracting JSON from: ", text)
        # Regular expression pattern to find JSON-like structures
        pattern = r'(\{.*?\}|\[.*?\])'
        
        potential_json_objects = re.findall(pattern, text, re.DOTALL)
        json_objects = []
        
        for obj_str in potential_json_objects:
            try:
                # remove any '[', ']'
                obj_str = obj_str.replace('[', '').replace(']', '')
                # Attempt to parse the JSON object string
                obj = json.loads(obj_str)
                json_objects.append(obj)
            except json.JSONDecodeError:
                # Not a valid JSON object, ignore
                pass
                
        # Return the last JSON containing x arguments
        for obj in json_objects[::-1]:
            if 'x' in str(obj):
                for key in obj.keys():
                    # Convert x indeces to latex compatible syntax, e.g., x1 -> x_1, x2 -> x_2, etc.
                    indeces = []
                    value_str = obj[key]
                    for index, char in enumerate(obj[key]):
                        if char == 'x' and index + 1 < len(value_str) and value_str[index + 1].isdigit():
                            indeces.append(index + 1)
                    for index in indeces[::-1]:
                        obj[key] = obj[key][:index] + '_' + obj[key][index:]
                return obj
            
__all__ = ['find_math_expressions', 'find_json_in_text']
