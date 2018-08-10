import pmf
import functools
import itertools
import operator


def die(number_of_faces: int):
    return pmf.ints([1 / number_of_faces] * number_of_faces, 1)


def d(*args):
    if len(args) == 1:
        return die(args[0])
    elif len(args) == 2:
        return functools.reduce(operator.__add__, itertools.repeat(die(args[1]), args[0]))
    else:
        raise ValueError(args)
