

def rekey_dict(input_dict):
    return {index + 1: value for index, value in enumerate(input_dict.values())}
