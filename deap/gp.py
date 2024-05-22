#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

"""The :mod:`gp` module provides the methods and classes to perform
Genetic Programming with DEAP. It essentially contains the classes to
build a Genetic Program Tree, and the functions to evaluate it.

This module support both strongly and loosely typed GP.
"""
import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
import ast
import sympy
import json
from inspect import isclass

import numpy

from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt

from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

from . import tools  # Needed by HARM-GP
from . import creator

######################################
# GP Data structure                  #
######################################

# Define the name of type for any types.
__type__ = object

# gpLocals = {
#     'sub': lambda x, y: x - y,
#     'protected_div': lambda x, y: x/y,
#     'mul': lambda x, y: x*y,
#     'add': lambda x, y: x + y,
#     'sin': lambda x: sympy.sin(x),
#     'cos': lambda x: sympy.cos(x),
#     'tan': lambda x: sympy.tan(x),
#     'protected_sqrt': lambda x: sympy.sqrt(x),
#     'protected_log': lambda x: sympy.ln(x),
#     'exp': lambda x: sympy.exp(x),
#     'sinh': lambda x: sympy.sinh(x),
#     'cosh': lambda x: sympy.cosh(x),
#     'tanh': lambda x: sympy.tanh(x),
#     'asin': lambda x: sympy.asin(x),
#     'acos': lambda x: sympy.acos(x),
#     'atan': lambda x: sympy.atan(x),
#     # 'abs': lambda x: sympy.Abs(x),
#     'asinh': lambda x: sympy.asinh(x),
#     'acosh': lambda x: sympy.acosh(x),
#     'atanh': lambda x: sympy.atanh(x),
# }

class PrimitiveTree(list):
    """Tree specifically formatted for optimization of genetic programming
    operations. The tree is represented with a list, where the nodes are
    appended, or are assumed to have been appended when initializing an object
    of this class with a list of primitives and terminals e.g. generated with
    the method **gp.generate**, in a depth-first order.
    The nodes appended to the tree are required to have an attribute *arity*,
    which defines the arity of the primitive. An arity of 0 is expected from
    terminals nodes.
    """

    def __init__(self, content):
        list.__init__(self, content)

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    def __setitem__(self, key, val):
        # Check for most common errors
        # Does NOT check for STGP constraints
        if isinstance(key, slice):
            if key.start >= len(self):
                raise IndexError("Invalid slice object (try to assign a %s"
                                 " in a tree of size %d). Even if this is allowed by the"
                                 " list object slice setter, this should not be done in"
                                 " the PrimitiveTree context, as this may lead to an"
                                 " unpredictable behavior for searchSubtree or evaluate."
                                 % (key, len(self)))
            total = val[0].arity
            for node in val[1:]:
                total += node.arity - 1
            if total != 0:
                raise ValueError("Invalid slice assignation : insertion of"
                                 " an incomplete subtree is not allowed in PrimitiveTree."
                                 " A tree is defined as incomplete when some nodes cannot"
                                 " be mapped to any position in the tree, considering the"
                                 " primitives' arity. For instance, the tree [sub, 4, 5,"
                                 " 6] is incomplete if the arity of sub is 2, because it"
                                 " would produce an orphan node (the 6).")
        elif val.arity != self[key].arity:
            raise ValueError("Invalid node replacement with a node of a"
                             " different arity.")
        list.__setitem__(self, key, val)

    def __str__(self):
        """Return the expression in a human readable string.
        """
        string = ""
        stack = []
        for node in self:
            stack.append((node, []))
            while len(stack[-1][1]) == stack[-1][0].arity:
                prim, args = stack.pop()
                string = prim.format(*args)
                if len(stack) == 0:
                    break  # If stack is empty, all nodes should have been seen
                stack[-1][1].append(string)

        return string

    @classmethod
    def from_string(cls, string, pset):
        """Try to convert a string expression into a PrimitiveTree given a
        PrimitiveSet *pset*. The primitive set needs to contain every primitive
        present in the expression.

        :param string: String representation of a Python expression.
        :param pset: Primitive set from which primitives are selected.
        :returns: PrimitiveTree populated with the deserialized primitives.
        """
        tokens = re.split("[ \t\n\r\f\v(),]", string)
        expr = []
        ret_types = deque()
        for token in tokens:
            if token == '':
                continue
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

            if token in pset.mapping:
                primitive = pset.mapping[token]

                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError("Primitive {} return type {} does not "
                                    "match the expected one: {}."
                                    .format(primitive, primitive.ret, type_))

                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            else:
                try:
                    token = eval(token)
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))

                if type_ is None:
                    type_ = type(token)

                if not issubclass(type(token), type_):
                    raise TypeError("Terminal {} type {} does not "
                                    "match the expected one: {}."
                                    .format(token, type(token), type_))

                expr.append(Terminal(token, False, type_))
        return cls(expr)

    @property
    def height(self):
        """Return the height of the tree, or the depth of the
        deepest node.
        """
        stack = [0]
        max_depth = 0
        for elem in self:
            depth = stack.pop()
            max_depth = max(max_depth, depth)
            stack.extend([depth + 1] * elem.arity)
        return max_depth

    @property
    def root(self):
        """Root of the tree, the element 0 of the list.
        """
        return self[0]

    def searchSubtree(self, begin):
        """Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        end = begin + 1
        total = self[begin].arity
        while total > 0:
            total += self[end].arity - 1
            end += 1
        return slice(begin, end)


class Primitive(object):
    """Class that encapsulates a primitive and when called with arguments it
    returns the Python code to call the primitive with the arguments.

        >>> pr = Primitive("mul", (int, int), int)
        >>> pr.format(1, 2)
        'mul(1, 2)'
    """
    __slots__ = ('name', 'arity', 'args', 'ret', 'seq')

    def __init__(self, name, args, ret):
        self.name = name
        self.arity = len(args)
        self.args = args
        self.ret = ret
        args = ", ".join(map("{{{0}}}".format, range(self.arity)))
        self.seq = "{name}({args})".format(name=self.name, args=args)

    def format(self, *args):
        return self.seq.format(*args)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class Terminal(object):
    """Class that encapsulates terminal primitive in expression. Terminals can
    be values or 0-arity functions.
    """
    __slots__ = ('name', 'value', 'ret', 'conv_fct')

    def __init__(self, terminal, symbolic, ret):
        self.ret = ret
        self.value = terminal
        self.name = str(terminal)
        self.conv_fct = str if symbolic else repr

    @property
    def arity(self):
        return 0

    def format(self):
        return self.conv_fct(self.value)

    def __eq__(self, other):
        if type(self) is type(other):
            return all(getattr(self, slot) == getattr(other, slot)
                       for slot in self.__slots__)
        else:
            return NotImplemented


class MetaEphemeral(type):
    """Meta-Class that creates a terminal which value is set when the
    object is created. To mutate the value, a new object has to be
    generated.
    """
    cache = {}

    def __new__(meta, name, func, ret=__type__, id_=None):
        if id_ in MetaEphemeral.cache:
            return MetaEphemeral.cache[id_]

        if isinstance(func, types.LambdaType) and func.__name__ == '<lambda>':
            warnings.warn("Ephemeral {name} function cannot be "
                          "pickled because its generating function "
                          "is a lambda function. Use functools.partial "
                          "instead.".format(name=name), RuntimeWarning)

        def __init__(self):
            self.value = func()

        attr = {'__init__': __init__,
                'name': name,
                'func': func,
                'ret': ret,
                'conv_fct': repr}

        cls = super(MetaEphemeral, meta).__new__(meta, name, (Terminal,), attr)
        MetaEphemeral.cache[id(cls)] = cls
        return cls

    def __init__(cls, name, func, ret=__type__, id_=None):
        super(MetaEphemeral, cls).__init__(name, (Terminal,), {})

    def __reduce__(cls):
        return (MetaEphemeral, (cls.name, cls.func, cls.ret, id(cls)))


copyreg.pickle(MetaEphemeral, MetaEphemeral.__reduce__)


class PrimitiveSetTyped(object):
    """Class that contains the primitives that can be used to solve a
    Strongly Typed GP problem. The set also defined the researched
    function return type, and input arguments type and number.
    """

    def __init__(self, name, in_types, ret_type, prefix="ARG"):
        self.terminals = defaultdict(list)
        self.primitives = defaultdict(list)
        self.arguments = []
        # setting "__builtins__" to None avoid the context
        # being polluted by builtins function when evaluating
        # GP expression.
        self.context = {"__builtins__": None}
        self.mapping = dict()
        self.terms_count = 0
        self.prims_count = 0

        self.name = name
        self.ret = ret_type
        self.ins = in_types
        for i, type_ in enumerate(in_types):
            arg_str = "{prefix}{index}".format(prefix=prefix, index=i)
            self.arguments.append(arg_str)
            term = Terminal(arg_str, True, type_)
            self._add(term)
            self.terms_count += 1

    def renameArguments(self, **kargs):
        """Rename function arguments with new names from *kargs*.
        """
        for i, old_name in enumerate(self.arguments):
            if old_name in kargs:
                new_name = kargs[old_name]
                self.arguments[i] = new_name
                self.mapping[new_name] = self.mapping[old_name]
                self.mapping[new_name].value = new_name
                del self.mapping[old_name]

    def _add(self, prim):
        def addType(dict_, ret_type):
            if ret_type not in dict_:
                new_list = []
                for type_, list_ in dict_.items():
                    if issubclass(type_, ret_type):
                        for item in list_:
                            if item not in new_list:
                                new_list.append(item)
                dict_[ret_type] = new_list

        addType(self.primitives, prim.ret)
        addType(self.terminals, prim.ret)

        self.mapping[prim.name] = prim
        if isinstance(prim, Primitive):
            for type_ in prim.args:
                addType(self.primitives, type_)
                addType(self.terminals, type_)
            dict_ = self.primitives
        else:
            dict_ = self.terminals

        for type_ in dict_:
            if issubclass(prim.ret, type_):
                dict_[type_].append(prim)

    def addPrimitive(self, primitive, in_types, ret_type, name=None):
        """Add a primitive to the set.

        :param primitive: callable object or a function.
        :param in_types: list of primitives arguments' type
        :param ret_type: type returned by the primitive.
        :param name: alternative name for the primitive instead
                     of its __name__ attribute.
        """
        if name is None:
            name = primitive.__name__
        prim = Primitive(name, in_types, ret_type)

        assert name not in self.context or \
               self.context[name] is primitive, \
               "Primitives are required to have a unique name. " \
               "Consider using the argument 'name' to rename your " \
               "second '%s' primitive." % (name,)

        self._add(prim)
        self.context[prim.name] = primitive
        self.prims_count += 1

    def addTerminal(self, terminal, ret_type, name=None):
        """Add a terminal to the set. Terminals can be named
        using the optional *name* argument. This should be
        used : to define named constant (i.e.: pi); to speed the
        evaluation time when the object is long to build; when
        the object does not have a __repr__ functions that returns
        the code to build the object; when the object class is
        not a Python built-in.

        :param terminal: Object, or a function with no arguments.
        :param ret_type: Type of the terminal.
        :param name: defines the name of the terminal in the expression.
        """
        symbolic = False
        if name is None and callable(terminal):
            name = terminal.__name__

        assert name not in self.context, \
            "Terminals are required to have a unique name. " \
            "Consider using the argument 'name' to rename your " \
            "second %s terminal." % (name,)

        if name is not None:
            self.context[name] = terminal
            terminal = name
            symbolic = True
        elif terminal in (True, False):
            # To support True and False terminals with Python 2.
            self.context[str(terminal)] = terminal

        prim = Terminal(terminal, symbolic, ret_type)
        self._add(prim)
        self.terms_count += 1

    def addEphemeralConstant(self, name, ephemeral, ret_type):
        """Add an ephemeral constant to the set. An ephemeral constant
        is a no argument function that returns a random value. The value
        of the constant is constant for a Tree, but may differ from one
        Tree to another.

        :param name: name used to refers to this ephemeral type.
        :param ephemeral: function with no arguments returning a random value.
        :param ret_type: type of the object returned by *ephemeral*.
        """
        if name not in self.mapping:
            class_ = MetaEphemeral(name, ephemeral, ret_type)
        else:
            class_ = self.mapping[name]
            if class_.func is not ephemeral:
                raise Exception("Ephemerals with different functions should "
                                "be named differently, even between psets.")
            if class_.ret is not ret_type:
                raise Exception("Ephemerals with the same name and function "
                                "should have the same type, even between psets.")

        self._add(class_)
        self.terms_count += 1

    def addADF(self, adfset):
        """Add an Automatically Defined Function (ADF) to the set.

        :param adfset: PrimitiveSetTyped containing the primitives with which
                       the ADF can be built.
        """
        prim = Primitive(adfset.name, adfset.ins, adfset.ret)
        self._add(prim)
        self.prims_count += 1

    @property
    def terminalRatio(self):
        """Return the ratio of the number of terminals on the number of all
        kind of primitives.
        """
        return self.terms_count / float(self.terms_count + self.prims_count)


class PrimitiveSet(PrimitiveSetTyped):
    """Class same as :class:`~deap.gp.PrimitiveSetTyped`, except there is no
    definition of type.
    """

    def __init__(self, name, arity, prefix="ARG"):
        args = [__type__] * arity
        PrimitiveSetTyped.__init__(self, name, args, __type__, prefix)

    def addPrimitive(self, primitive, arity, name=None):
        """Add primitive *primitive* with arity *arity* to the set.
        If a name *name* is provided, it will replace the attribute __name__
        attribute to represent/identify the primitive.
        """
        assert arity > 0, "arity should be >= 1"
        args = [__type__] * arity
        PrimitiveSetTyped.addPrimitive(self, primitive, args, __type__, name)

    def addTerminal(self, terminal, name=None):
        """Add a terminal to the set."""
        PrimitiveSetTyped.addTerminal(self, terminal, __type__, name)

    def addEphemeralConstant(self, name, ephemeral):
        """Add an ephemeral constant to the set."""
        PrimitiveSetTyped.addEphemeralConstant(self, name, ephemeral, __type__)

def fromSympyStr(container, string, pset, possiblePset, constants):
        """Try to convert a python mathematical expression into a 
        PrimitiveTree given a PrimitiveSet *pset*. If the primitive 
        is not already in the set, it will be added. 

        :param container: PrimitiveTree or Individual class to be filled.
        :param string: String representation of a Python expression.
        :param pset: Primitive set from which primitives are selected.
        :param possiblePset: Primitive set from which primitives can be added.
        (to support adding primitives on the fly, and restrict it to a set of primitives)
        :param constants: Possible constants that can be added to the primitive set.
        notice that if llm generates a constant that is not in the possible constants,
        there will be an ERROR.
        :returns: PrimitiveTree populated with the deserialized primitives.
        """
        def dfs(node):
            if isinstance(node, ast.Constant):
                return [node.n]
            elif isinstance(node, ast.Name):
                return [node.id]
            elif isinstance(node, ast.BinOp):
                left = dfs(node.left)
                right = dfs(node.right)
                if isinstance(node.op, ast.Add):
                    return ['add'] + left + right
                elif isinstance(node.op, ast.Sub):
                    return ['sub'] + left + right
                elif isinstance(node.op, ast.Mult):
                    return ['mul'] + left + right
                elif isinstance(node.op, ast.Div):
                    return ['protected_div'] + left + right
                elif isinstance(node.op, ast.Pow):
                    # Exponent is an integer
                    if isinstance(node.right, ast.Constant) and isinstance(node.right.n, int) and node.right.n > 0:
                        result = left
                        for _ in range(node.right.n - 1):
                            result = ['mul'] + result + left
                        return result 
                    # Exponent is a constant positive float
                    elif isinstance(node.right, ast.Constant) and isinstance(node.right.n, float):
                        return ['exp', 'mul'] + right + ['protected_log'] + left
                    # Exponent is a variable
                    elif isinstance(node.right, ast.Name):
                        return ['exp', 'mul'] + right + ['protected_log'] + left
                    # Exponent is in nested form
                    elif (not isinstance(node.right, ast.Constant) and not isinstance(node.right, ast.Name)):
                        # Exponent is a negative nested form
                        if (isinstance(node.right, ast.UnaryOp)):
                            result = left
                            # Exponent is a negative const integer in the form -b = sub(0, b)
                            if (isinstance(node.right.operand, ast.Constant) and isinstance(node.right.operand.n, int)):
                                if isinstance(node.right.op, ast.USub) and isinstance(node.right.operand.n, int):
                                    for _ in range(node.right.operand.n - 1):
                                        result = ['mul'] + result + left
                                    return ['protected_div', 1] + result
                            # Any other nested unary operation is treated as: a**-b = exp(-1 *（b）* ln(a))
                            else:
                                return ['exp', 'mul'] + right + ['protected_log'] + left
                        # Exponent is a positive nested form
                        else:
                            return ['exp', 'mul'] + right + ['protected_log'] + left
                    else:
                        raise ValueError(
                            "Unsupported power operation: ", ast.dump(node))
            elif isinstance(node, ast.UnaryOp):
                operand = dfs(node.operand)
                if isinstance(node.op, ast.USub):
                    return ['mul', -1] + operand
                else:
                    raise ValueError("Unsupported unary operation")
            elif isinstance(node, ast.Call):
                func_name = (node.func.id if isinstance(
                    node.func, ast.Name) else ast.dump(node.func))
                args = sum([dfs(arg) for arg in node.args], [])
                if func_name == 'exp':
                    return ['exp'] + args
                elif func_name == 'sin':
                    return ['sin'] + args
                elif func_name == 'cos':
                    return ['cos'] + args
                elif func_name == 'tan':
                    return ['tan'] + args
                elif func_name == 'sinh':
                    return ['sinh'] + args
                elif func_name == 'cosh':
                    return ['cosh'] + args
                elif func_name == 'tanh':
                    return ['tanh'] + args
                elif func_name == 'sqrt':
                    return ['protected_sqrt'] + args
                elif func_name == 'log':
                    return ['protected_log'] + args
                elif func_name == 'asin':
                    return ['asin'] + args
                elif func_name == 'acos':
                    return ['acos'] + args
                elif func_name == 'atan':
                    return ['atan'] + args
                # elif func_name == 'Abs' or func_name == 'abs':
                #     return ['abs'] + args
                elif func_name == 'asinh':
                    return ['asinh'] + args
                elif func_name == 'acosh':
                    return ['acosh'] + args
                elif func_name == 'atanh':
                    return ['atanh'] + args
                else:
                    raise ValueError(f"Unsupported function: {func_name}")
            else:
                raise ValueError("Unsupported operation: ", ast.dump(node))
        
        try:
            transformations = (standard_transformations + 
                               (implicit_multiplication_application,) + (convert_xor,))
            string = parse_expr(string, transformations=transformations)
            # Remove all '_' from string for ast
            string = str(string).replace('_', '')
            tokens = dfs(ast.parse(string).body[0].value)

        except ValueError as e:
            raise ValueError("Unable to parse expression: " + string + "\n" + str(e))
        except Exception as e:
            raise Exception("Unable to parse expression: " + string + "\n" + str(e))

        expr = []
        ret_types = deque()
        for token in tokens:
            if token == '':
                continue
            if token == 'x': # For the initialization, since we don't go through our realtime parser
                token = 'x1'
            if len(ret_types) != 0:
                type_ = ret_types.popleft()
            else:
                type_ = None

            if str(token) in pset.mapping:
                primitive = pset.mapping[str(token)]

                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError("Primitive {} return type {} does not "
                                    "match the expected one: {}."
                                    .format(primitive, primitive.ret, type_))

                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            elif token in possiblePset or isinstance(token, int) or isinstance(token, float):
                if token in possiblePset:
                    pset.addPrimitive(possiblePset[token][0], possiblePset[token][1])
                else:
                    pset.addTerminal(token)
                
                primitive = pset.mapping[str(token)] 

                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError("Primitive {} return type {} does not "
                                    "match the expected one: {}."
                                    .format(primitive, primitive.ret, type_))

                expr.append(primitive)
                if isinstance(primitive, Primitive):
                    ret_types.extendleft(reversed(primitive.args))
            # Token is a famous constant
            elif token in constants:
                pset.addTerminal(constants[token], name=token)
                primitive = pset.mapping[str(token)]

                if type_ is not None and not issubclass(primitive.ret, type_):
                    raise TypeError("Primitive {} return type {} does not "
                                    "match the expected one: {}."
                                    .format(primitive, primitive.ret, type_))
                
                expr.append(primitive)
            # # Sometimes the llm or sympy uses E instead of e
            # elif token == 'E':
            #     pset.addTerminal(math.e, name=token)
            #     primitive = pset.mapping[token]
            else:
                try:
                    token = eval(token)
                except NameError:
                    raise TypeError("Unable to evaluate terminal: {}.".format(token))

                if type_ is None:
                    type_ = type(token)

                if not issubclass(type(token), type_):
                    raise TypeError("Terminal {} type {} does not "
                                    "match the expected one: {}."
                                    .format(token, type(token), type_))

                expr.append(Terminal(token, False, type_))
        return container(expr)

######################################
# GP Tree compilation functions      #
######################################
def compile(expr, pset):
    """Compile the expression *expr*.

    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param pset: Primitive set against which the expression is compile.
    :returns: a function if the primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    code = str(expr)
    if len(pset.arguments) > 0:
        # This section is a stripped version of the lambdify
        # function of SymPy 0.6.6.
        args = ",".join(arg for arg in pset.arguments)
        code = "lambda {args}: {code}".format(args=args, code=code)
    try:
        return eval(code, pset.context, {})
    except MemoryError:
        _, _, traceback = sys.exc_info()
        raise MemoryError("DEAP : Error in tree evaluation :"
                          " Python cannot evaluate a tree higher than 90. "
                          "To avoid this problem, you should use bloat control on your "
                          "operators. See the DEAP documentation for more information. "
                          "DEAP will now abort.").with_traceback(traceback)


def compileADF(expr, psets):
    """Compile the expression represented by a list of trees. The first
    element of the list is the main tree, and the following elements are
    automatically defined functions (ADF) that can be called by the first
    tree.


    :param expr: Expression to compile. It can either be a PrimitiveTree,
                 a string of Python code or any object that when
                 converted into string produced a valid Python code
                 expression.
    :param psets: List of primitive sets. Each set corresponds to an ADF
                  while the last set is associated with the expression
                  and should contain reference to the preceding ADFs.
    :returns: a function if the main primitive set has 1 or more arguments,
              or return the results produced by evaluating the tree.
    """
    adfdict = {}
    func = None
    for pset, subexpr in reversed(list(zip(psets, expr))):
        pset.context.update(adfdict)
        func = compile(subexpr, pset)
        adfdict.update({pset.name: func})
    return func


######################################
# GP Program generation functions    #
######################################
def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has the same depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A full tree with all leaves at the same depth.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height."""
        return depth == height

    return generate(pset, min_, max_, condition, type_)


def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth
    between *min* and *max*.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths.
    """

    def condition(height, depth):
        """Expression generation stops when the depth is equal to height
        or when it is randomly determined that a node should be a terminal.
        """
        return depth == height or \
            (depth >= min_ and random.random() < pset.terminalRatio)

    return generate(pset, min_, max_, condition, type_)


def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*.
    Half the time, the expression is generated with :func:`~deap.gp.genGrow`,
    the other half, the expression is generated with :func:`~deap.gp.genFull`.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)


def genRamped(pset, min_, max_, type_=None):
    """
    .. deprecated:: 1.0
        The function has been renamed. Use :func:`~deap.gp.genHalfAndHalf` instead.
    """
    warnings.warn("gp.genRamped has been renamed. Use genHalfAndHalf instead.",
                  FutureWarning)
    return genHalfAndHalf(pset, min_, max_, type_)


def generate(pset, min_, max_, condition, type_=None):
    """Generate a tree as a list of primitives and terminals in a depth-first
    order. The tree is built from the root to the leaves, and it stops growing
    the current branch when the *condition* is fulfilled: in which case, it
    back-tracks, then tries to grow another branch until the *condition* is
    fulfilled again, and so on. The returned list can then be passed to the
    constructor of the class *PrimitiveTree* to build an actual tree object.

    :param pset: Primitive set from which primitives are selected.
    :param min_: Minimum height of the produced trees.
    :param max_: Maximum Height of the produced trees.
    :param condition: The condition is a function that takes two arguments,
                      the height of the tree to build and the current
                      depth in the tree.
    :param type_: The type that should return the tree when called, when
                  :obj:`None` (default) the type of :pset: (pset.ret)
                  is assumed.
    :returns: A grown tree with leaves at possibly different depths
              depending on the condition function.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    height = random.randint(min_, max_)
    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        if condition(height, depth):
            try:
                term = random.choice(pset.terminals[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a terminal of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            if type(term) is MetaEphemeral:
                term = term()
            expr.append(term)
        else:
            try:
                prim = random.choice(pset.primitives[type_])
            except IndexError:
                _, _, traceback = sys.exc_info()
                raise IndexError("The gp.generate function tried to add "
                                 "a primitive of type '%s', but there is "
                                 "none available." % (type_,)).with_traceback(traceback)
            expr.append(prim)
            for arg in reversed(prim.args):
                stack.append((depth + 1, arg))
    return expr

def llmUpdateInd(llm, func, toReplace, pset):
    """Gets new llm individuals that are not in llmIndividuals.

    :param llm: The llm object containing the llm individuals and the llm chat
    context.
    :param func: The function/constructor use to create the new individuals.
    :param toReplace: A dictionary mapping the index of the individual to replace
                        to the individual itself.
    :param pset: Primitive set to check whether it has changed after creating new
                    individuals.
    """
    toReplaceList = []

    # Construct a json matching expression with fitness
    for key in toReplace:
        try:
            # sympyExpr = sympy.sympify(str(toReplace[key]), locals=tools.sympyLocals)
            sympyExpr = tools.safeSympify(str(toReplace[key]))
        except Exception as e:
            print("Error when converting to sympy expression in llmUpdateInd: ", e)
            sympyExpr = ""
        fitness = ""
        if not toReplace[key].fitness.valid:
            fitness = "Not evaluated"
        else:
            fitness = str(toReplace[key].fitness.values[0])

        toReplaceList.append({str(sympyExpr): fitness})
        # toReplaceJson.update({str(sympyExpr): fitness})

    # Convert to string
    # toReplaceJson = json.dumps(toReplaceJson)
    toReplaceList = str(toReplaceList)

    # Format arguments properly
    arguments = ""
    for arg in range(llm.dimensions):
        arguments += "x" + str(arg+1) + ", "
    # Remove extra comma
    arguments = arguments[:-2]

    message=("For a given Genetic Programming algorithm for a " +
            str(llm.dimensions) + "-dimensional (with " + arguments + " as argument(s)) "
            "Symbolic Regression problem, "
            "the following are some random individuals from our population, "
            "together with their corresponding Root Mean Squared Error (RMSE) "
            "values if they are known: "
            + toReplaceList + ". "
            "According to the best individuals (lowest RMSE), and to "
            "well-known Symbolic Regression benchmarks, "
            "replace all of these with examples of other unique "
            "genetic programming "
            "individuals in the form of compact mathematical expressions. "
            "Your final answer must be in the form of a single JSON object, with the key "
            "being the individual number (from '1' to '" + str(len(toReplace)) + "'), "
            "and the value being the mathematical expression. Your expressions don't need "
            "to be perfectly accurate.")
    
    message = message.replace('[', '')
    message = message.replace(']', '')

    # user = ("The followng is the RMSE (the lower the better) results of using different "
    #         "regression models to fit the data: " + llm.regressions + " "
    #         "These regressions are a strong indicatif of the type of function we are "
    #         "looking for.")
    # assistant = ("A lower RMSE means the expression we are looking for is in a similar form "
    #         "to the regression model.")
    # llm.initialAnalysis(user, assistant)

    # user = ("The followng are random individuals from our GP population together with their fitness: "
    #             + toReplaceList + ". ")
    # assistant = ("A lower RMSE fitness score means the expression we are looking for is "
    #                 "potentially a good candidate, and should consider similar forms.")
    # llm.initialAnalysis(user, assistant)

    # inds = str(len(toReplace))
    # message = ("The following individuals are already in the population: " + llm.individuals +
    #             ". Estimate " +
    #             inds + " new GP individuals, that are not in this population. "
    #             "Half of your individuals should be similar to the good performing ones, and the other "
    #             "half according to well-known Symbolic Regression benchmarks (e.g., Koza, Keijzer, "
    #             "RatPol3D, Nguyen etc.), or real-world problems. "
    #             "(e.g., Newton's law, Ohm's law, heat transfer, breeder's equation, etc.) "
    #             "Your answer must be in JSON with simple python mathematical expressions, and without "
    #             "any comments or explanations.")
    #             # "{'1': 'expression1', '2': ..., '" + inds + "': 'expression" + inds + "'}.")

    # Get new individuals
    # newInds = llm.addQuestion(message).content
    newInds = llm.complete(message)

    # Convert to mapping
    # newInds = json.loads(newInds)
    try:
        newInds = tools.find_json_in_text(newInds)
    except Exception as e:
        raise Exception("Unable to parse JSON in llmUpdateInd: " + str(e))

    # Convert to list of values
    newInds = list(newInds.values())

    # Add the new individuals to the llm individuals json?
    inds = json.loads(llm.individuals)
    j = len(inds)
    for i in range(len(newInds)):
        inds[str(j + i + 1)] = newInds[i]

    # Convert to trees
    llmContainer = []
    oldPset = pset
    while j < len(inds):
        try:
            st = str(inds[str(j+1)])
            ind = func(string=st)
            llmContainer.append(ind)
            llm.libraryAdd(ind, st)
            j = j + 1
        except ValueError as e:
            print("A value error occured in llmUpdate parsing the expression: ", inds[str(j+1)])
            print("Error: ", e)
            print("Skipping this individual (one less individual will be replaced)")
            del inds[str(j+1)]
            j = j + 1
            continue
        except TypeError as e:
            print("A type error occured in llmUpdate parsing the expression: ", inds[str(j+1)])
            print("Error: ", e)
            print("Skipping this individual (one less individual will be replaced)")
            del inds[str(j+1)]
            j = j + 1
            continue
        except Exception as e:
            print("An error occured in llmUpdate parsing the expression: ", inds[str(j+1)])
            print("Error: ", e)
            print("Skipping this individual (one less individual will be replaced)")
            del inds[str(j+1)]
            j = j + 1
            continue
    
    # Update the llm individuals
    llm.individuals = json.dumps(inds)

    # Check if the pset has changed, if so, update probabilities
    # if oldPset != pset:
        # Do llmUpdateProbabilities
    
    # return the new individuals
    return llmContainer

def llamaUpdate(expr, candidates, llm, func):
    """Tries to replace expr with another individual generated by the LLM

    :param func: The function/constructor use to create the new individuals.
    :param candidates: Random individuals from population for LLM to get priors
    :param expr: Individual to replace
    """

    try:
        sympyExpr = tools.safeSympify(str(expr))
    except Exception as e:
        raise Exception("Failed converting expr: " + str(expr) +
                        " to sympy expression in llamaUpdate: " + str(e))

    # Convert to string
    tempExpr = str(sympyExpr)

    candidatesList = []

    # Construct a json matching expression with fitness
    for key in candidates:
        try:
            # sympyExpr = sympy.sympify(str(candidates[key]), locals=tools.sympyLocals)
            sympyExpr = tools.safeSympify(str(candidates[key]))
        except Exception as e:
            print("Failed converting to sympy expression in llmUpdateInd: ", e)
            sympyExpr = "Unknown"
        fitness = ""
        if not candidates[key].fitness.valid:
            fitness = "Not evaluated"
        else:
            fitness = str(candidates[key].fitness.values[0])

        candidatesList.append({str(sympyExpr): fitness})
        # candidatesJson.update({str(sympyExpr): fitness})

    # Convert to string
    # candidatesJson = json.dumps(candidatesJson)
    candidatesList = str(candidatesList)

    # Format arguments properly
    arguments = ""
    for arg in range(llm.dimensions):
        arguments += "x" + str(arg+1) + ", "
    # Remove extra comma
    arguments = arguments[:-2]

    message=("For a given Genetic Programming algorithm for a " +
            str(llm.dimensions) + "-dimensional (with " + arguments + " as argument(s)) "
            "Symbolic Regression problem, "
            "the following are some random individuals from our population, "
            "together with their corresponding Root Mean Squared Error (RMSE) "
            "values if they are known: "
            + candidatesList + ". "
            "According to the best individuals (lowest RMSE), and to "
            "well-known Symbolic Regression benchmarks or real-world problems, "
            "replace the function " + tempExpr + " with the estimate of another new, "
            "different and unique genetic programming individual in the form of a "
            "mathematical expression.")

    # Get the new individual
    newInd = llm.complete(message)
    
    # Find mathematical expression in the text
    newInd = tools.find_math_expressions(newInd)

    # Convert to tree
    try:
        # st = str(newInd).replace('^', '**')
        newInd = func(string=str(newInd))
    except Exception as e:
        raise Exception("Failed converting new individual: " + str(newInd) +
                        " to tree in llamaUpdate: " + str(e))
    
    return newInd,

def llamaSimplify(expr, func, llm):
    """Tries to simplify the given individual

    :param expre: The expression to symplify
    :param func: The function/constructor use to create the new individuals.
    """

    try:
        sympyExpr = tools.safeSympify(str(expr))
    except Exception as e:
        raise Exception("Failed converting: " + str(expr) +
                        " to sympy expression in llamaSimplify: " + str(e))

    # Convert to string
    tempExpr = str(sympyExpr)

    try:
        float(tempExpr)
        return expr,
    except ValueError:
        pass

    message =("Give a compact representation of the function below. "
        "In order to shorten the final function as much as possible, "
        "nested combinations "
        "of trigonometric, logarithmic or exponential operations must "
        "be approximated. "
        + tempExpr)
    
    # Get the simplified individual
    simplifiedInd = llm.complete(message)
    
    # Find mathematical expression in the text
    simplifiedInd = tools.find_math_expressions(simplifiedInd)

    # Convert to tree
    try:
        # st = str(simplifiedInd).replace('^', '**')
        simplifiedInd = func(string=str(simplifiedInd))
    except Exception as e:
        raise Exception("Failed converting simplified individual: " + str(simplifiedInd) +
                        " to tree in llamaSimplify: " + str(e))
    
    return simplifiedInd,

def llmSimplify(llm, func, func2, toSimplify, best, bestI):
    """Simplify the given individuals using the llm.

    :param llm: The llm object containing the llm individuals and the llm chat context.
    :param func: The function/constructor use to create the new individuals.
    :param func2: Fail safe function/constructor use to create new individuals in case
                    the first one fails. (Because the main algorithm expects precisely
                    the number of individuals to be simplified and in case where 
                    len(llmIndividuals) != len(toSimplify), the wrong individual
                    will be simplified)
    :param toSimplify: A dictionary mapping the index of the individual to simplify
                        to the individual itself.
    :param best: The best individual in the population.
    :param bestI: The index of the best individual in the population.
    """
    llm.reset()

    toSimplifyString = {} # Mapping of the index of the individual to simplify to the expression

    # Construct a list of expressions to simplify
    for key in toSimplify:
        try:
            sympyExpr = tools.safeSympify(str(toSimplify[key]))
            # sympyExpr = sympy.sympify(str(toSimplify[key]), locals=tools.sympyLocals)
            if not toSimplify[key].fitness.valid:
                Warning("Individual to simplify has invalid fitness.")
        except Exception as e:
            print("Error when converting to sympy expression in llmSimplify: ", e)
            sympyExpr = ""
        
        toSimplifyString.update({key: str(sympyExpr)})
    
    # Convert best individual to string
    try:
        best = tools.safeSympify(str(best))
        # best = sympy.sympify(str(best), locals=tools.sympyLocals)
    except Exception as e:
        print("Error when converting best to sympy expression in llmSimplify: ", e)
        best = ""

    toSimplifyString.update({bestI: str(best)})

    first = str(next(iter(toSimplifyString)))

    # last = str(bestI) # Last one will be the best individual (added later)
    last = str(next(reversed(toSimplifyString)))

    # Convert to string
    toSimplifyString = str(toSimplifyString)
    
    question = ("The followng are some random individuals and the best one from our "
                "GP population: " + toSimplifyString + ". "
                "With the best individual in the population being: " + str(best) + ". "
                "Please try to simplify them using common mathematical "
                "rules and identities. Your final answer must be in JSON with the "
                "simplified sympy mathematical expressions, and without any comments or "
                "explanations."
                "In case you are unable to simplify an expression, please return the "
                "original one: " 
                "{'" + first + "': 'expression', ... , '" + last + "': 'expression'}.")

    # Get the simplified individuals
    simplifiedInd_s = llm.addQuestion(question).content

    # Convert to mapping
    simplifiedInds = json.loads(simplifiedInd_s)

    # Prepare the container
    llmContainer = {}

    # Convert to trees
    for key in simplifiedInds:
        if not key.isnumeric():
            continue
        if int(key) not in toSimplify and int(key) != bestI:
            print("The LLM returned an individual that was not in the list of individuals"
                "to simplify. Using fail safe function to replace")
            # simplifiedInds[key] = func2()
            llmContainer.update({key: [func2(), True]})
            continue
        try:
            st = str(simplifiedInds[key]).replace('^', '**')
            # llmContainer.append({key: func(string=st)})
            # simplifiedInds[key] = func(string=st)
            llmContainer.update({key: [func(string=st), False]})
        except ValueError as e:
            print("A value error occured in llmSimplify parsing the expression: ", st)
            print("Error: ", e)
            print("Using fail safe function to create new individual instead")
            # simplifiedInds[key] = func2()
            llmContainer.update({key: [func2(), True]})
            continue
        except TypeError as e:
            print("A type error occured in llmSimplify parsing the expression: ", st)
            print("Error: ", e)
            print("Using fail safe function to create new individual instead")
            # simplifiedInds[key] = func2()
            llmContainer.update({key: [func2(), True]})
            continue
        except Exception as e:
            print("An error occured in llmSimplify parsing the expression: ", st)
            print("Error: ", e)
            print("Using fail safe function to create new individual instead")
            # simplifiedInds[key] = func2()
            llmContainer.update({key: [func2(), True]})
            continue
    # return the updated individuals
    return llmContainer

######################################
# GP Crossovers                      #
######################################

def cxOnePoint(ind1, ind2):

    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = list(range(1, len(ind1)))
        types2[__type__] = list(range(1, len(ind2)))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


def cxOnePointLeafBiased(ind1, ind2, termpb):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First typed tree participating in the crossover.
    :param ind2: Second typed tree participating in the crossover.
    :param termpb: The probability of choosing a terminal node (leaf).
    :returns: A tuple of two typed trees.

    When the nodes are strongly typed, the operator makes sure the
    second node type corresponds to the first node type.

    The parameter *termpb* sets the probability to choose between a terminal
    or non-terminal crossover point. For instance, as defined by Koza, non-
    terminal primitives are selected for 90% of the crossover points, and
    terminals for 10%, so *termpb* should be set to 0.1.
    """

    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # Determine whether to keep terminals or primitives for each individual
    terminal_op = partial(eq, 0)
    primitive_op = partial(lt, 0)
    arity_op1 = terminal_op if random.random() < termpb else primitive_op
    arity_op2 = terminal_op if random.random() < termpb else primitive_op

    # List all available primitive or terminal types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)

    for idx, node in enumerate(ind1[1:], 1):
        if arity_op1(node.arity):
            types1[node.ret].append(idx)

    for idx, node in enumerate(ind2[1:], 1):
        if arity_op2(node.arity):
            types2[node.ret].append(idx)

    common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        # Set does not support indexing
        type_ = random.sample(common_types, 1)[0]
        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


######################################
# GP Mutations                       #
######################################
def mutUniform(individual, expr, pset):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    slice_ = individual.searchSubtree(index)
    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_)
    return individual,


def mutNodeReplacement(individual, pset):
    """Replaces a randomly chosen primitive from *individual* by a randomly
    chosen primitive with the same number of arguments from the :attr:`pset`
    attribute of the individual.

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    if len(individual) < 2:
        return individual,

    index = random.randrange(1, len(individual))
    node = individual[index]

    if node.arity == 0:  # Terminal
        term = random.choice(pset.terminals[node.ret])
        if type(term) is MetaEphemeral:
            term = term()
        individual[index] = term
    else:  # Primitive
        prims = [p for p in pset.primitives[node.ret] if p.args == node.args]
        individual[index] = random.choice(prims)

    return individual,


def mutEphemeral(individual, mode):
    """This operator works on the constants of the tree *individual*. In
    *mode* ``"one"``, it will change the value of one of the individual
    ephemeral constants by calling its generator function. In *mode*
    ``"all"``, it will change the value of **all** the ephemeral constants.

    :param individual: The normal or typed tree to be mutated.
    :param mode: A string to indicate to change ``"one"`` or ``"all"``
                 ephemeral constants.
    :returns: A tuple of one tree.
    """
    if mode not in ["one", "all"]:
        raise ValueError("Mode must be one of \"one\" or \"all\"")

    ephemerals_idx = [index
                      for index, node in enumerate(individual)
                      if isinstance(type(node), MetaEphemeral)]

    if len(ephemerals_idx) > 0:
        if mode == "one":
            ephemerals_idx = (random.choice(ephemerals_idx),)

        for i in ephemerals_idx:
            individual[i] = type(individual[i])()

    return individual,


def mutInsert(individual, pset):
    """Inserts a new branch at a random position in *individual*. The subtree
    at the chosen position is used as child node of the created subtree, in
    that way, it is really an insertion rather than a replacement. Note that
    the original subtree will become one of the children of the new primitive
    inserted, but not perforce the first (its position is randomly selected if
    the new primitive has more than one child).

    :param individual: The normal or typed tree to be mutated.
    :returns: A tuple of one tree.
    """
    index = random.randrange(len(individual))
    node = individual[index]
    slice_ = individual.searchSubtree(index)
    choice = random.choice

    # As we want to keep the current node as children of the new one,
    # it must accept the return value of the current node
    primitives = [p for p in pset.primitives[node.ret] if node.ret in p.args]

    if len(primitives) == 0:
        return individual,

    new_node = choice(primitives)
    new_subtree = [None] * len(new_node.args)
    position = choice([i for i, a in enumerate(new_node.args) if a == node.ret])

    for i, arg_type in enumerate(new_node.args):
        if i != position:
            term = choice(pset.terminals[arg_type])
            if isclass(term):
                term = term()
            new_subtree[i] = term

    new_subtree[position:position + 1] = individual[slice_]
    new_subtree.insert(0, new_node)
    individual[slice_] = new_subtree
    return individual,


def mutShrink(individual):
    """This operator shrinks the *individual* by choosing randomly a branch and
    replacing it with one of the branch's arguments (also randomly chosen).

    :param individual: The tree to be shrunk.
    :returns: A tuple of one tree.
    """
    # We don't want to "shrink" the root
    if len(individual) < 3 or individual.height <= 1:
        return individual,

    iprims = []
    for i, node in enumerate(individual[1:], 1):
        if isinstance(node, Primitive) and node.ret in node.args:
            iprims.append((i, node))

    if len(iprims) != 0:
        index, prim = random.choice(iprims)
        arg_idx = random.choice([i for i, type_ in enumerate(prim.args) if type_ == prim.ret])
        rindex = index + 1
        for _ in range(arg_idx + 1):
            rslice = individual.searchSubtree(rindex)
            subtree = individual[rslice]
            rindex += len(subtree)

        slice_ = individual.searchSubtree(index)
        individual[slice_] = subtree

    return individual,


######################################
# GP bloat control decorators        #
######################################


def staticLimit(key, max_value):
    """Implement a static limit on some measurement on a GP tree, as defined
    by Koza in [Koza1989]. It may be used to decorate both crossover and
    mutation operators. When an invalid (over the limit) child is generated,
    it is simply replaced by one of its parents, randomly selected.

    This operator can be used to avoid memory errors occurring when the tree
    gets higher than 90 levels (as Python puts a limit on the call stack
    depth), because it can ensure that no tree higher than this limit will ever
    be accepted in the population, except if it was generated at initialization
    time.

    :param key: The function to use in order the get the wanted value. For
                instance, on a GP tree, ``operator.attrgetter('height')`` may
                be used to set a depth limit, and ``len`` to set a size limit.
    :param max_value: The maximum value allowed for the given measurement.
    :returns: A decorator that can be applied to a GP operator using \
    :func:`~deap.base.Toolbox.decorate`

    .. note::
       If you want to reproduce the exact behavior intended by Koza, set
       *key* to ``operator.attrgetter('height')`` and *max_value* to 17.

    .. [Koza1989] J.R. Koza, Genetic Programming - On the Programming of
        Computers by Means of Natural Selection (MIT Press,
        Cambridge, MA, 1992)

    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            keep_inds = [copy.deepcopy(ind) for ind in args]
            new_inds = list(func(*args, **kwargs))
            for i, ind in enumerate(new_inds):
                if not isinstance(ind, str):
                    if key(ind) > max_value:
                        new_inds[i] = random.choice(keep_inds)
            return new_inds

        return wrapper

    return decorator


######################################
# GP bloat control algorithms        #
######################################

def harm(population, toolbox, cxpb, mutpb, ngen,
         alpha, beta, gamma, rho, nbrindsmodel=-1, mincutoff=20,
         stats=None, halloffame=None, verbose=__debug__):
    """Implement bloat control on a GP evolution using HARM-GP, as defined in
    [Gardner2015]. It is implemented in the form of an evolution algorithm
    (similar to :func:`~deap.algorithms.eaSimple`).

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param alpha: The HARM *alpha* parameter.
    :param beta: The HARM *beta* parameter.
    :param gamma: The HARM *gamma* parameter.
    :param rho: The HARM *rho* parameter.
    :param nbrindsmodel: The number of individuals to generate in order to
                            model the natural distribution. -1 is a special
                            value which uses the equation proposed in
                            [Gardner2015] to set the value of this parameter :
                            max(2000, len(population))
    :param mincutoff: The absolute minimum value for the cutoff point. It is
                        used to ensure that HARM does not shrink the population
                        too much at the beginning of the evolution. The default
                        value is usually fine.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. note::
       The recommended values for the HARM-GP parameters are *alpha=0.05*,
       *beta=10*, *gamma=0.25*, *rho=0.9*. However, these parameters can be
       adjusted to perform better on a specific problem (see the relevant
       paper for tuning information). The number of individuals used to
       model the natural distribution and the minimum cutoff point are less
       important, their default value being effective in most cases.

    .. [Gardner2015] M.-A. Gardner, C. Gagne, and M. Parizeau, Controlling
        Code Growth by Dynamically Shaping the Genotype Size Distribution,
        Genetic Programming and Evolvable Machines, 2015,
        DOI 10.1007/s10710-015-9242-8

    """

    def _genpop(n, pickfrom=[], acceptfunc=lambda s: True, producesizes=False):
        # Generate a population of n individuals, using individuals in
        # *pickfrom* if possible, with a *acceptfunc* acceptance function.
        # If *producesizes* is true, also return a list of the produced
        # individuals sizes.
        # This function is used 1) to generate the natural distribution
        # (in this case, pickfrom and acceptfunc should be let at their
        # default values) and 2) to generate the final population, in which
        # case pickfrom should be the natural population previously generated
        # and acceptfunc a function implementing the HARM-GP algorithm.
        producedpop = []
        producedpopsizes = []
        while len(producedpop) < n:
            if len(pickfrom) > 0:
                # If possible, use the already generated
                # individuals (more efficient)
                aspirant = pickfrom.pop()
                if acceptfunc(len(aspirant)):
                    producedpop.append(aspirant)
                    if producesizes:
                        producedpopsizes.append(len(aspirant))
            else:
                opRandom = random.random()
                if opRandom < cxpb:
                    # Crossover
                    aspirant1, aspirant2 = toolbox.mate(*map(toolbox.clone,
                                                             toolbox.select(population, 2)))
                    del aspirant1.fitness.values, aspirant2.fitness.values
                    if acceptfunc(len(aspirant1)):
                        producedpop.append(aspirant1)
                        if producesizes:
                            producedpopsizes.append(len(aspirant1))

                    if len(producedpop) < n and acceptfunc(len(aspirant2)):
                        producedpop.append(aspirant2)
                        if producesizes:
                            producedpopsizes.append(len(aspirant2))
                else:
                    aspirant = toolbox.clone(toolbox.select(population, 1)[0])
                    if opRandom - cxpb < mutpb:
                        # Mutation
                        aspirant = toolbox.mutate(aspirant)[0]
                        del aspirant.fitness.values
                    if acceptfunc(len(aspirant)):
                        producedpop.append(aspirant)
                        if producesizes:
                            producedpopsizes.append(len(aspirant))

        if producesizes:
            return producedpop, producedpopsizes
        else:
            return producedpop

    def halflifefunc(x):
        return x * float(alpha) + beta

    if nbrindsmodel == -1:
        nbrindsmodel = max(2000, len(population))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Estimation population natural distribution of sizes
        naturalpop, naturalpopsizes = _genpop(nbrindsmodel, producesizes=True)

        naturalhist = [0] * (max(naturalpopsizes) + 3)
        for indsize in naturalpopsizes:
            # Kernel density estimation application
            naturalhist[indsize] += 0.4
            naturalhist[indsize - 1] += 0.2
            naturalhist[indsize + 1] += 0.2
            naturalhist[indsize + 2] += 0.1
            if indsize - 2 >= 0:
                naturalhist[indsize - 2] += 0.1

        # Normalization
        naturalhist = [val * len(population) / nbrindsmodel for val in naturalhist]

        # Cutoff point selection
        sortednatural = sorted(naturalpop, key=lambda ind: ind.fitness)
        cutoffcandidates = sortednatural[int(len(population) * rho - 1):]
        # Select the cutoff point, with an absolute minimum applied
        # to avoid weird cases in the first generations
        cutoffsize = max(mincutoff, len(min(cutoffcandidates, key=len)))

        # Compute the target distribution
        def targetfunc(x):
            return (gamma * len(population) * math.log(2)
                    / halflifefunc(x)) * math.exp(-math.log(2)
                                                  * (x - cutoffsize) / halflifefunc(x))

        targethist = [naturalhist[binidx] if binidx <= cutoffsize else
                      targetfunc(binidx) for binidx in range(len(naturalhist))]

        # Compute the probabilities distribution
        probhist = [t / n if n > 0 else t for n, t in zip(naturalhist, targethist)]

        def probfunc(s):
            return probhist[s] if s < len(probhist) else targetfunc(s)

        def acceptfunc(s):
            return random.random() <= probfunc(s)

        # Generate offspring using the acceptance probabilities
        # previously computed
        offspring = _genpop(len(population), pickfrom=naturalpop,
                            acceptfunc=acceptfunc, producesizes=False)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def graph(expr):
    """Construct the graph of a tree expression. The tree expression must be
    valid. It returns in order a node list, an edge list, and a dictionary of
    the per node labels. The node are represented by numbers, the edges are
    tuples connecting two nodes (number), and the labels are values of a
    dictionary for which keys are the node numbers.

    :param expr: A tree expression to convert into a graph.
    :returns: A node list, an edge list, and a dictionary of labels.

    The returned objects can be used directly to populate a
    `pygraphviz <http://networkx.lanl.gov/pygraphviz/>`_ graph::

        import pygraphviz as pgv

        # [...] Execution of code that produce a tree expression

        nodes, edges, labels = graph(expr)

        g = pgv.AGraph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        g.layout(prog="dot")

        for i in nodes:
            n = g.get_node(i)
            n.attr["label"] = labels[i]

        g.draw("tree.pdf")

    or a `NetworX <http://networkx.github.com/>`_ graph::

        import matplotlib.pyplot as plt
        import networkx as nx

        # [...] Execution of code that produce a tree expression

        nodes, edges, labels = graph(expr)

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.graphviz_layout(g, prog="dot")

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.show()


    .. note::

       We encourage you to use `pygraphviz
       <http://networkx.lanl.gov/pygraphviz/>`_ as the nodes might be plotted
       out of order when using `NetworX <http://networkx.github.com/>`_.
    """
    nodes = list(range(len(expr)))
    edges = list()
    labels = dict()

    stack = []
    for i, node in enumerate(expr):
        if stack:
            edges.append((stack[-1][0], i))
            stack[-1][1] -= 1
        labels[i] = node.name if isinstance(node, Primitive) else node.value
        stack.append([i, node.arity])
        while stack and stack[-1][1] == 0:
            stack.pop()

    return nodes, edges, labels


######################################
# RDO Mutation                       #
######################################

class Library:
    """This class is used to store the subtrees with their semantics
    """
    exprPool = []
    points = []
    toolbox = []

    def __init__(self, toolbox, points):
        """
        :param toolbox: DEAP initialized toolbox
        :param points: training points
        """
        self.toolbox = toolbox
        self.points = points

    def similarity(self, sv1, sv2):
        """compare the similarity between two semantic vector"""
        return ((sv1-sv2)**2).sum()/sv1.size

    def libraryAdd(self, individual, st):
        """
        Add the subtree individual to the library. 
        i.e., key: [subTree, individual, string]

        :param individual: the subtree individual (Individual object)
        :param st: a sympy compatible expression"""
        # Check if st already in library, if so do not add again
        for i in range(len(self.exprPool)):
            if st == str(self.exprPool[i][2]):
                return
        # [0] is the SubTree object with semantics vector and [1] is the subtree individual
        self.exprPool.append([SubTree(self.toolbox, individual, self.points), individual, st])
    
    def librarySearch(self, tarSem):
        """
        Search the library for the best subtree. i.e., min(min(d(y, s(p)))) 
        where y is the target semantics, s(p) is the semantics of the subtree
        program p in the library
        
        :param tarSem: target semantics"""

        best = self.exprPool[0]
        for i in range(len(self.exprPool)):
            if self.similarity(tarSem, self.exprPool[i][0].sem_vec) < \
                self.similarity(tarSem, best[0].sem_vec):

                best = self.exprPool[i]

        return best

class SubTree(object):
    expr=[]
    sem_vec=[]

    def __init__(self, toolbox, expr_st, points):
        """expr_st: the sub tree individual (Individual object)"""
        self.expr = expr_st
        func = toolbox.compile(expr=expr_st)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x
        # sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
        sqerrors = []
        for x in points:
            try:
                sqerrors.append(func(*x))
                # sqerrors.append(func(x[0]))
            except:
                sqerrors.append(500)

        sqerrors = numpy.array(sqerrors)

        self.sem_vec=sqerrors
        #print(self.sem_vec)

def decode(ind):
    """
    ind: type individual
    return: parent list (each item: Node type (primitive/terminal, index))
    children list (Node type)
    """
    parent= [] #1 dimension list
    children= []  #2 dimension list
    pi=-1   #pi: the processing index;   usi: the last unsatisfied index
    parent.append(pi)
    pi = usi = pi + 1
    if len(ind) == 1:
        return parent, children

    for i in range(1, len(ind)):

        #update the parent
        parent.append(pi)    #record the parent of current node
        #pi = pi + 1          #update the parent

        #update the children
        if pi + 1 > len(children):  # if the parent node is new, append new item to the tail of children
            children.append([i])

        # if the parent node has been met, append the item to the existing item of children
        else:
            children[pi].append(i)

        #update the usi
        #if pi is satisfied
        if ind[pi].arity == len(children[pi]):
            #find the last unsatisfied parent
            while ind[usi].arity <= len(children[usi]):
                if usi >= 1:
                    usi = usi - 1
                else:
                    break
        #if pi hasn't been satisfied
        else:
            usi = pi

        #update the pi
        #if i is the primitive, pi follows i
        if ind[i].arity > 0:
            pi = usi = i

        #else if i is the terminal / ephemeral && pi is not satisfied, pi keeps still
        #else if i is the terminal / ephemeral && pi is satisfied, pi follows usi
        elif ind[i].arity==0 and ind[pi].arity == len(children[pi]):
            pi = usi

        #if i is the terminal / ephemeral, insert the placeholder into children
        if ind[i].arity == 0:
            children.append([-1,-1])

    return parent, children

def llmRandomDesiredOperator(par, llm, tarSem, toolbox, points):
    """
    Performs llm optimization of the subtree in addition to the RDO from
    [Semantic Backpropagation for Designing Search Operators in Genetic ProgrammingPawlak et al., 2013]

    :param par: individual to mutate    
    :param llm: LLM object which contains the library of subtrees
    :param tarSem: Target semantics
    :param toolbox: DEAP initialized toolbox
    :param points: training points
    """

    # Select a random node in the tree
    index = random.randrange(len(par))
    slice_ = par.searchSubtree(index)

    # Get the desired semantics of the subtree rooted at 'index'
    dsrSem = semBackpropagation(toolbox, points, tarSem, par, index)

    # Search the library for the best subtree
    subtree = llm.librarySearch(dsrSem)

    # Optimize the subtree with the llm
    # Choose random points from sem_vec
    semPoints = random.sample(range(len(subtree[0].sem_vec)), 5)

    x = [points[i] for i in semPoints]
    subtreeOut = [subtree[0].sem_vec[i] for i in semPoints] # Output we have
    tarOut = [dsrSem[i] for i in semPoints] # Output we want
    func = subtree[2] # String of the function      
    error = ""  
    if func.find("x") != -1:
        # llm.reset()
        # question = ("For the points: " + str(x) + ". The function " + func +
        #             " returns " + str(subtreeOut) + ". Make some small adjustments "
        #             "to the function so that its output is closer to " + str(tarOut) + ". "
        #             "Your answer should be in JSON format, with '1' as the key. "
        #             "You must not provide additional comments or explanations.")
        # updatedFunc = llm.addQuestion(question).content
        question = ("For these values of x: "
                    + str(x) + ". "
                    "The function "
                    + func + " "
                    "returns "
                    + str(subtreeOut) + " "
                    "respectively. Make some adjustments "
                    "to the function so that its output is closer to "
                    + str(tarOut))
        try:
            completion = llm.complete(question)
            updatedFunc = tools.llmparser.find_math_expressions(completion)
            # updatedFunc = json.loads(updatedFunc) # old
            if updatedFunc != func:
                optimizedSubtree = toolbox.fromLLM(string=updatedFunc)
                ind = creator.Individual(optimizedSubtree)
                optimizedSubtree = SubTree(toolbox, ind, points)
                llm.libraryAdd(ind, updatedFunc)
                subtree = [optimizedSubtree, ind, updatedFunc]            
        except:
            # We don't really care why it failed
            error = ("The optimization of the subtree failed. Subtree not updated")
    
    # Add the slice of the selected subtree to the library if its height is less than 4
    # ind = creator.Individual(par[slice_])
    # if ind.height < 4:
    #     st = str(ind)
    #     sympyExpr = sympy.sympify(st, locals=tools.sympyLocals)
    #     sympyExpr = str(sympyExpr)
    #     llm.libraryAdd(ind, sympyExpr)

    # Replace the subtree with the optimized subtree
    par[slice_] = subtree[1]

    return par, error,
    
def semBackpropagation(toolbox, points, tarSem, par, node):
    """
    Semantic backpropagation algorithm for tree programs

    :param toolbox: DEAP initialized toolbox
    :param points: training points
    :param tarSem: Target semantics
    :param par: program tree
    :param node: the node in p to be reached by the backpropagation

    :return: D, the desired semantics of the subtree rooted at 'node'. 
    """

    # D = [None] * len(tarSem)
    # for i in range(len(tarSem)):
    #     d = tarSem[i]
    #     a = par.root
    #     while a != node:
    #         k = a.arity
    #         dp = [None] * len(d)
    #         for o in d:
    #             dp = dp + invert(a, k, o)
    #         d = dp
    #         a = child(a, node)
    #     D[i] = d
    # return D

    # decode the individual
    parent_arr, children_arr = decode(par)

    # Identify the path from root to the selected node
    path = [node]
    travel = node
    while parent_arr[travel] != -1:
        path.append(parent_arr[travel])
        travel = parent_arr[travel]
    path.reverse()

    # based on the children_arr, construct the other subtree and get its semantic
    dsrSem = tarSem
    if len(path) > 1:
        for j in range(1, len(path)):
            if len(children_arr[parent_arr[j]]) == 1:
                dsrSem = invert(par[parent_arr[j]], 0, dsrSem)
            else:
                # index of the selected node in left subtree
                if j == children_arr[parent_arr[j]][0]:
                    # calculate the semantic of the right subtree
                    fixSt = children_arr[parent_arr[j]][1]
                    subTree = SubTree(toolbox, creator.Individual(par[par.searchSubtree(fixSt)]), points)
                    dsrSem = invert(par[parent_arr[j]], 0, dsrSem, subSem1=subTree.sem_vec)
                # index of the selected node in right subtree
                else:
                    # calculate the semantic of the left subtree
                    fixSt = children_arr[parent_arr[j]][0]
                    subTree = SubTree(toolbox, creator.Individual(par[par.searchSubtree(fixSt)]), points)
                    dsrSem = invert(par[parent_arr[j]], 1, dsrSem, subSem0=subTree.sem_vec)
    # print(dsrSem)
    return dsrSem

def invert(con, k, tarSem, subSem0=numpy.zeros(1), subSem1=numpy.zeros(1)):
    """
    Invert the instruction node con (a) with k children and target semantic(s) tarSem (o)

    con: the node content (a in the paper)
    k: the index of param 0 / 1
    tarSem: desired semantic (o in the paper)
    """
    dsr_sem=numpy.zeros(tarSem.size)
    if con.name == "add":
        if k==0:
            dsr_sem = tarSem - subSem1
        else:
            dsr_sem = tarSem - subSem0
    elif con.name=="sub":
        if k==0:
            dsr_sem = tarSem + subSem1
        else:
            dsr_sem = subSem0 - tarSem
    if con.name == "mul":
        if k==0:
            for i in range(len(tarSem)):
                if subSem1[i]!=0:
                    dsr_sem[i] = tarSem[i] / subSem1[i]
                if subSem1[i]==0 and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                if subSem1[i]==0 and tarSem[i]!=0:
                    dsr_sem[i] = 0
        else:
            for i in range(len(tarSem)):
                if subSem0[i]!=0:
                    dsr_sem[i] = tarSem[i] / subSem0[i]
                if subSem0[i]==0 and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                if subSem0[i]==0 and tarSem[i]!=0:
                    dsr_sem[i] = 0

    if con.name =="protected_div":
        if k==0:
            for i in range(len(tarSem)):
                if math.isfinite(subSem1[i]):
                    dsr_sem[i] = tarSem[i]*subSem1[i]
                if math.isinf(subSem1[i]) and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                if math.isinf(subSem1[i]) and tarSem[i]!=0:
                    dsr_sem[i] = 0
        else:
            for i in range(len(tarSem)):
                if subSem0[i]!=0 and tarSem[i]!=0:
                    dsr_sem[i] = subSem0[i]/tarSem[i]
                elif subSem0[i]==0 and tarSem[i]==0:
                    dsr_sem[i] = tarSem[i]
                elif subSem0[i]==0 and tarSem[i]!=0:
                    dsr_sem[i] = 0
    if con.name =="protected_sqrt":
        for i in range(len(tarSem)):
            if tarSem[i]>=0:
                dsr_sem[i] = tarSem[i]**2
            else:
                dsr_sem[i] = 0
    if con.name == "sin":
        for i in range(len(tarSem)):
            if tarSem[i] > 1 or tarSem[i] < -1:
                dsr_sem[i] = tarSem[i]
            else:
                dsr_sem[i] = math.asin(tarSem[i])
    if con.name == "cos":
        for i in range(len(tarSem)):
            if tarSem[i] > 1 or tarSem[i] < -1:
                dsr_sem[i] = tarSem[i]
            else:
                dsr_sem[i] = math.acos(tarSem[i])
    if con.name == "tan":
        for i in range(len(tarSem)):
            if tarSem[i] > 1 or tarSem[i] < -1:
                dsr_sem[i] = tarSem[i]
            else:
                dsr_sem[i] = math.atan(tarSem[i])
    if con.name == "sinh":
        for i in range(len(tarSem)):
            dsr_sem[i] = math.asinh(tarSem[i])
    if con.name == "cosh":
        for i in range(len(tarSem)):
            if tarSem[i] < 1:
                dsr_sem[i] = 1
            else:
                dsr_sem[i] = math.acosh(tarSem[i])
    if con.name == "tanh":
        for i in range(len(tarSem)):
            if tarSem[i] >= 1 or tarSem[i] <= -1:
                dsr_sem[i] = tarSem[i]
            else:
                dsr_sem[i] = math.atanh(tarSem[i])
    if con.name == "asin":
        for i in range(len(tarSem)):
            dsr_sem[i] = math.sin(tarSem[i])
    if con.name == "acos":
        for i in range(len(tarSem)):
            if tarSem[i] > 1 or tarSem[i] < -1:
                dsr_sem[i] = tarSem[i]
            else:
                dsr_sem[i] = math.cos(tarSem[i])
    if con.name == "atan":
        for i in range(len(tarSem)):
            dsr_sem[i] = math.tan(tarSem[i])
    # if con.name == "abs":
    #     for i in range(len(tarSem)):
    #         if tarSem[i] < 0:
    #             dsr_sem[i] = -tarSem[i]
    #         else:
    #             dsr_sem[i] = tarSem[i]
    if con.name == "asinh":
        for i in range(len(tarSem)):
            if tarSem[i] < 700 and tarSem[i] > -700:
                dsr_sem[i] = math.sinh(tarSem[i])
            else:
                dsr_sem[i] = tarSem[i]
    if con.name == "acosh":
        for i in range(len(tarSem)):
            if tarSem[i] < 700 and tarSem[i] > -700:
                dsr_sem[i] = math.cosh(tarSem[i])
            else:
                dsr_sem[i] = tarSem[i]
    if con.name == "atanh":
        for i in range(len(tarSem)):
            dsr_sem[i] = math.tanh(tarSem[i])
    if con.name == "exp":
        for i in range(len(tarSem)):
            if tarSem[i] <= 0:
                dsr_sem[i] = tarSem[i]
            else:
                dsr_sem[i] = math.log(tarSem[i])
    if con.name == "protected_log":
        for i in range(len(tarSem)):
            if tarSem[i] > 10:
                tarSem[i] = 10
            dsr_sem[i] = math.exp(tarSem[i])

    return dsr_sem
        
    
######################################
# GSGP Mutation                      #
######################################            

def mutSemantic(individual, gen_func=genGrow, pset=None, ms=None, min=2, max=6):
    """
    Implementation of the Semantic Mutation operator. [Geometric semantic genetic programming, Moraglio et al., 2012]
    mutated_individual = individual + logistic * (random_tree1 - random_tree2)

    :param individual: individual to mutate
    :param gen_func: function responsible for the generation of the random tree that will be used during the mutation
    :param pset: Primitive Set, which contains terminal and operands to be used during the evolution
    :param ms: Mutation Step
    :param min: min depth of the random tree
    :param max: max depth of the random tree
    :return: mutated individual

    The mutated contains the original individual

        >>> import operator
        >>> def lf(x): return 1 / (1 + math.exp(-x));
        >>> pset = PrimitiveSet("main", 2)
        >>> pset.addPrimitive(operator.sub, 2)
        >>> pset.addTerminal(3)
        >>> pset.addPrimitive(lf, 1, name="lf")
        >>> pset.addPrimitive(operator.add, 2)
        >>> pset.addPrimitive(operator.mul, 2)
        >>> individual = genGrow(pset, 1, 3)
        >>> mutated = mutSemantic(individual, pset=pset, max=2)
        >>> ctr = sum([m.name == individual[i].name for i, m in enumerate(mutated[0])])
        >>> ctr == len(individual)
        True
    """
    for p in ['lf', 'mul', 'add', 'sub']:
        assert p in pset.mapping, "A '" + p + "' function is required in order to perform semantic mutation"

    tr1 = gen_func(pset, min, max)
    tr2 = gen_func(pset, min, max)
    # Wrap mutation with a logistic function
    tr1.insert(0, pset.mapping['lf'])
    tr2.insert(0, pset.mapping['lf'])
    if ms is None:
        ms = random.uniform(0, 2)
    mutation_step = Terminal(ms, False, object)
    # Create the root

    new_ind = individual
    new_ind.insert(0, pset.mapping["add"])
    # Append the left branch
    new_ind.append(pset.mapping["mul"])
    new_ind.append(mutation_step)
    new_ind.append(pset.mapping["sub"])
    # Append the right branch
    new_ind.extend(tr1)
    new_ind.extend(tr2)

    return new_ind,


def cxSemantic(ind1, ind2, gen_func=genGrow, pset=None, min=2, max=6):
    """
    Implementation of the Semantic Crossover operator [Geometric semantic genetic programming, Moraglio et al., 2012]
    offspring1 = random_tree1 * ind1 + (1 - random_tree1) * ind2
    offspring2 = random_tree1 * ind2 + (1 - random_tree1) * ind1

    :param ind1: first parent
    :param ind2: second parent
    :param gen_func: function responsible for the generation of the random tree that will be used during the mutation
    :param pset: Primitive Set, which contains terminal and operands to be used during the evolution
    :param min: min depth of the random tree
    :param max: max depth of the random tree
    :return: offsprings

    The mutated offspring contains parents

        >>> import operator
        >>> def lf(x): return 1 / (1 + math.exp(-x));
        >>> pset = PrimitiveSet("main", 2)
        >>> pset.addPrimitive(operator.sub, 2)
        >>> pset.addTerminal(3)
        >>> pset.addPrimitive(lf, 1, name="lf")
        >>> pset.addPrimitive(operator.add, 2)
        >>> pset.addPrimitive(operator.mul, 2)
        >>> ind1 = genGrow(pset, 1, 3)
        >>> ind2 = genGrow(pset, 1, 3)
        >>> new_ind1, new_ind2 = cxSemantic(ind1, ind2, pset=pset, max=2)
        >>> ctr = sum([n.name == ind1[i].name for i, n in enumerate(new_ind1)])
        >>> ctr == len(ind1)
        True
        >>> ctr = sum([n.name == ind2[i].name for i, n in enumerate(new_ind2)])
        >>> ctr == len(ind2)
        True
    """
    for p in ['lf', 'mul', 'add', 'sub']:
        assert p in pset.mapping, "A '" + p + "' function is required in order to perform semantic crossover"

    tr = gen_func(pset, min, max)
    tr.insert(0, pset.mapping['lf'])
    new_ind1 = ind1
    new_ind1.insert(0, pset.mapping["mul"])
    new_ind1.insert(0, pset.mapping["add"])
    new_ind1.extend(tr)
    new_ind1.append(pset.mapping["mul"])
    new_ind1.append(pset.mapping["sub"])
    new_ind1.append(Terminal(1.0, False, object))
    new_ind1.extend(tr)
    new_ind1.extend(ind2)

    new_ind2 = ind2
    new_ind2.insert(0, pset.mapping["mul"])
    new_ind2.insert(0, pset.mapping["add"])
    new_ind2.extend(tr)
    new_ind2.append(pset.mapping["mul"])
    new_ind2.append(pset.mapping["sub"])
    new_ind2.append(Terminal(1.0, False, object))
    new_ind2.extend(tr)
    new_ind2.extend(ind1)

    return new_ind1, new_ind2


if __name__ == "__main__":
    import doctest

    doctest.testmod()
