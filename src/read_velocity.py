

from traveltime import read_NLLvel


def load_model(file, format):

    if format.upper() == "NLL":
        velocity_model = read_NLLvel(file)
    else:
        raise ValueError("Unknown velocity format: ", format)

    return velocity_model





