import argparse


def get_name():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_name',type=str)
    
    return parser.parse_args()
