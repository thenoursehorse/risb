from copy import deepcopy

def load_history(x, error, max_size):
    if (len(x) != len(error)) and (len(x) != (len(error) + 1)):
        raise ValueError('x and error are the wrong lengths !')

    x_out = deepcopy(x)
    error_out = deepcopy(error)

    while len(x_out) >= max_size:
        x_out.pop()
    while len(error_out) >= max_size:
        error_out.pop()

    return x_out, error_out

def insert_vector(vec, vec_new, max_size=None):
    # Note these operations are mutable on input list
    vec.insert(0, vec_new)
    if max_size is not None:
        if len(vec) >= max_size:
            vec.pop()