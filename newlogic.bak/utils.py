from itertools import zip_longest

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def grouper_variable_length(iterable, n):
    "Collect data into variable-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF G"
    block = []
    for val in iterable:
        block.append(val)
        if len(block) >= n:
            yield block
            block = []
    if len(block) >= 1:
        yield block