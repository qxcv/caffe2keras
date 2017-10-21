import string
import functools
from keras.layers import *  # noqa
from keras.models import Model  # noqa

indent = 4
ret_model = 'ret_model'


def varname(obj):
    '''varname makes variable names from keras / tensorflow objects'''
    n = obj.name.replace(':', '_')
    slash_index = string.find(n, '/')
    if slash_index >= 0:
        n = n[:slash_index]
    return n


class CodeGenerator(object):
    '''CodeGenerator object is a passthrough for printing code to generate the
       objects, while actaully generating the objects.'''

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
                print >> self.f, self.s + '(%s)' % str(v)

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
                return varname(prev)

    def __init__(self, filename):
        self.f = None

        if filename is not None:
            imports = """from keras.layers import *  # noqa
from keras.models import Model  # noqa"""

            self.f = open(filename, "w")
            print >> self.f, imports
            print >> self.f
            print >> self.f
            print >> self.f, "def make_model():"
            print >> self.f

    @staticmethod
    def keras(nodes):
        '''keras provides access to the underlying keras object. This is only used
           for model construction by the convert script'''
        try:
            return [n.obj for n in nodes]
        except:
            return nodes.obj

    def invoked(self, a, *args, **kwargs):
        '''invoked is the general passthrough layer that creates the object
           creation string and then generats the StringFunctor based on it'''

        # Create the object
        s = "{}(".format(a)

        # Format the args on line per arg

        sargs = ',\n'.join([repr(arg) for arg in args]).replace('?', 'None')
        skwargs = ',\n'.join([str(x) + '=' + repr(kwargs[x]) for x in kwargs.keys()]).replace('?', 'None')

        # put the string form of args and kwargs in the braces
        s += ',\n'.join([x for x in [sargs, skwargs] if x != '']).lstrip()
        s += ')'

        # generate the object
        obj = eval(s)

        # get a variable name for the object (this will be what the script things it's called
        varset = varname(obj) + ' = '

        # re-indent the whole thing based on the line '<indent>varname = Object('
        sset = s.split('\n')
        prestr = ''.join(' ' for _ in range(indent))
        s = prestr + varset + sset[0]
        prestr = ''.join(' ' for _ in range(string.find(s, '(') + 1))
        srest = [prestr + snext for snext in sset[1:]]
        s = '\n'.join([s] + srest)

        return CodeGenerator.StringFunctor(self.f, obj, s)

    def __getattr__(self, a):
        '''__getattr__ for any method binds the 'invoked' to that method'''
        return functools.partial(self.invoked, a)

    def Input(self, *args, **kwargs):  # noqa
        '''Input needs a special override - Input needs to get printed at
           node create time, not call time (Inputs are the beginning of the graph
           and don't get called with any previous object'''
        sf = self.invoked('Input', *args, **kwargs)
        # Input is special, there is no before to activate him
        if self.f is not None:
            print >> self.f, sf.s
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
            print >> self.f, s
        return Model(**kwargs)

    def close(self):
        '''close finishes the file and closes it.'''
        if self.f is not None:
            print >> self.f, ''.join(' ' for _ in range(indent)) + "return %s" % ret_model
            self.f.close()
