from prospect import prospector_argparser
from prospect import run_prospector


def build_obs(**run_params):
    """
    """
    return obs


def build_model(**run_params):
    """
    """
    return mod


def build_sps(**run_params):
    """
    """
    return sps



def build_all(**kwargs):
    return (build_obs(**kwargs), bulid_model(**kwargs),
            build_sps(**kwargs), build_noise(**kwargs))

if __name__=='__main__':
    parser = prospector_argparser.get_parser()  # parser with default arguments 
    parser.add_argument('--custom_argument_1', ...)
    parser.add_argument('--custom_argument_2', ...)
    
    args = parser.parse_args()
    run_params = vars(args)

    obs, mod, sps, noise = build_all(**run_params)
    fit_model(obs, mod, sps, noise=noise)
