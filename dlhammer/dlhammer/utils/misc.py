# -*- coding: utf-8 -*-
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================

import ast


def merge_dict(a, b, path=None):
    """merge b into a. The values in b will override values in a.

    Args:
        a (dict): dict to merge to.
        b (dict): dict to merge from.

    Returns: dict1 with values merged from b.

    """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a


def merge_opts(d, opts):
    """merge opts
    Args:
        d (dict): The dict.
        opts (list): The opts to merge. format: [key1, name1, key2, name2,...]
    Returns: d. the input dict `d` with merged opts.

    """
    assert len(opts) % 2 == 0, f'length of opts must be even. Got: {opts}'
    for i in range(0, len(opts), 2):
        full_k, v = opts[i], opts[i + 1]
        keys = full_k.split('.')
        sub_d = d
        for i, k in enumerate(keys):
            if not hasattr(sub_d, k):
                raise ValueError(f'The key {k} not exist in the dict. Full key:{full_k}')
            if i != len(keys) - 1:
                sub_d = sub_d[k]
            else:
                sub_d[k] = v
    return d


def to_string(params, indent=2):
    """format params to a string

    Args:
        params (EasyDict): the params. 

    Returns: The string to display.

    """
    msg = '{\n'
    for i, (k, v) in enumerate(params.items()):
        if isinstance(v, dict):
            v = to_string(v, indent + 4)
        spaces = ' ' * indent
        msg += spaces + '{}: {}'.format(k, v)
        if i == len(params) - 1:
            msg += ' }'
        else:
            msg += '\n'
    return msg


def eval_dict_leaf(d):
    """eval values of dict leaf.

    Args:
        d (dict): The dict to eval.

    Returns: dict.

    """
    for k, v in d.items():
        if not isinstance(v, dict):
            d[k] = eval_string(v)
        else:
            eval_dict_leaf(v)
    return d


def eval_string(string):
    """automatically evaluate string to corresponding types.
    
    For example:
        not a string  -> return the original input
        '0'  -> 0
        '0.2' -> 0.2
        '[0, 1, 2]' -> [0,1,2]
        'eval(1+2)' -> 3
        'eval(range(5))' -> [0,1,2,3,4]


    Args:
        value : string.

    Returns: the corresponding type

    """
    if not isinstance(string, str):
        return string
    if len(string) > 1 and string[0] == '[' and string[-1] == ']':
        return eval(string)
    if string[0:5] == 'eval(':
        return eval(string[5:-1])
    try:
        v = ast.literal_eval(string)
    except:
        v = string
    return v
