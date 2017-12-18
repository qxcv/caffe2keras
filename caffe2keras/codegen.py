from __future__ import print_function
import re
import types
import inspect
import functools
from keras.layers import *  # noqa
from .extra_layers import Scale  # noqa
from keras.models import Model
from keras import backend as K  # noqa

indent = 4
ret_model = 'ret_model'
max_var_len = 12


def varname(obj):
    '''varname makes variable names from keras / tensorflow objects'''
    n = obj.name.replace(':', '_C_').replace('-', '__')
    slash_index = n.find('/')
    if slash_index >= 0:
        n = n[:slash_index]
    return n


class RawRepr(object):
    def __init__(self, out):
        self.out = out

    def __repr__(self):
        return self.out


class CodeGenerator(object):
    '''CodeGenerator object is a passthrough for printing code to generate the
       objects, while actaully generating the objects.'''
    _varname_registry = {}
    _var_count = 0

    class StringFunctor(object):
        '''StringFunctor is a callable object that prints itself when called.

           This is so we can do things like Layer(args)(previous)'''
        def __init__(self, f, obj, s):
            self.f = f      # file
            self.obj = obj  # instanciated object
            self.s = s      # string to instanciate object

        def __call__(self, prev):

            if self.f is not None:
                # Convert the passed in argument to its variable name for printing
                v = self.to_varname(prev)
                print(self.s + '(%s)' % str(v), file=self.f)

            # These may be keras, tensorflow, or StringFunctors. Need to get to the
            # raw object to do the call
            if isinstance(prev, CodeGenerator.StringFunctor):
                prev = prev.obj
            return self.obj(prev)

        @staticmethod
        def to_varname(prev):
            '''to_varname takes object or list of objects and converts to their printable
               variable names'''
            try:
                return '[' + ', '.join([CodeGenerator.StringFunctor.to_varname(x) for x in prev]) + ']'
            except:
                if isinstance(prev, CodeGenerator.StringFunctor):
                    return CodeGenerator.StringFunctor.to_varname(prev.obj)
                return CodeGenerator.varname(prev)

    @staticmethod
    def varname(vn):
        vn = varname(vn)
        if len(vn) > max_var_len:
            if vn in CodeGenerator._varname_registry:
                vn = CodeGenerator._varname_registry[vn]
            else:
                vns = vn[:max_var_len] + '_' + "%.04d" % CodeGenerator._var_count
                CodeGenerator._var_count += 1
                CodeGenerator._varname_registry[vn] = vns
                vn = vns
        return vn

    def __init__(self, filename):
        self.f = None
        # XXX This is really brittle. At least balance the parens.
        self.lambda_re = re.compile("Lambda\(([^)]*)\)")

        if filename is not None:
            imports = """from keras.layers import *  # noqa
from keras.models import Model
from keras import backend as K  # noqa
try:
    from caffe2keras.extra_layers.Scale import Scale #noqa
except:
    print("find the layer Scale")
"""

            self.f = open(filename, "w")
            print(imports, file=self.f)
            print(file=self.f)
            print(file=self.f)
            print("def make_model():", file=self.f)
            print(file=self.f)

    @staticmethod
    def keras(nodes):
        '''keras provides access to the underlying keras object. This is only used
           for model construction by the convert script'''
        try:
            return [n.obj for n in nodes]
        except:
            try:
                return nodes.obj
            except:
                return nodes

    def invoked(self, a, *pre_args, **kwargs):
        '''invoked is the general passthrough layer that creates the object
           creation string and then generats the StringFunctor based on it'''

        args = []
        for arg in pre_args:
            if isinstance(arg, types.FunctionType):
                codestr = inspect.getsource(arg.func_code)
                m = self.lambda_re.search(codestr)
                codestr = m.group(1)
                args.append(RawRepr(codestr))
            else:
                args.append(arg)

        # Create the object
        s = "{}(".format(a)

        # Format the args on line per arg

        sargs = ',\n'.join([repr(arg) for arg in args]).replace('?', 'None')
        skwargs = ',\n'.join([str(x) + '=' + repr(kwargs[x]) for x in kwargs.keys()]).replace('?', 'None')

        # put the string form of args and kwargs in the braces
        s += ',\n'.join([x for x in [sargs, skwargs] if x != ''])
        s += ')'

        # generate the object
        obj = eval(s)

        # get a variable name for the object (this will be what the script things it's called
        varset = CodeGenerator.varname(obj) + ' = '

        # re-indent the whole thing based on the line '<indent>varname = Object('
        sset = s.split('\n')
        prestr = ' ' * indent
        s = prestr + varset + sset[0]
        prestr = ' ' * (s.find('(') + 1)
        srest = [prestr + snext for snext in sset[1:]]
        s = '\n'.join([s] + srest)

        return CodeGenerator.StringFunctor(self.f, obj, s)

    def __getattr__(self, a):
        '''__getattr__ for any method binds the 'invoked' to that method'''
        return functools.partial(self.invoked, a)

    def LambdaStr(self, expr, *args, **kwargs):
        expr = RawRepr(expr)
        return self.invoked('Lambda', expr, *args, **kwargs)

    def Input(self, *args, **kwargs):  # noqa
        '''Input needs a special override - Input needs to get printed at
           node create time, not call time (Inputs are the beginning of the graph
           and don't get called with any previous object'''
        sf = self.invoked('Input', *args, **kwargs)
        # Input is special, there is no before to activate him
        if self.f is not None:
            print(sf.s, file=self.f)
        return sf

    def Model(self, *args, **kwargs):  # noqa
        '''Model needs a special override - Model takes lists of objects passed
           in to the create. These need to be formatted properly so we just do it
           in place here'''
        base = ''.join(' ' for _ in range(indent)) + "%s = Model(" % (ret_model)
        skwargs = [str(x) + '=' + CodeGenerator.StringFunctor.to_varname(kwargs[x]) for x in kwargs.keys()]
        prestr = ''.join(' ' for _ in range(len(base)))
        base += skwargs[0]
        srest = [prestr + snext for snext in skwargs[1:]]
        s = ',\n'.join([base] + srest) + ')'

        if self.f is not None:
            print(s, file=self.f)
        return Model(**kwargs)

    def close(self):
        '''close finishes the file and closes it.'''
        if self.f is not None:
            print(''.join(' ' for _ in range(indent)) + "return %s" % ret_model, file=self.f)
            self.f.close()
