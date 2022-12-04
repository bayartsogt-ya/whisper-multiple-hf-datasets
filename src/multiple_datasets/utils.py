import pandas as pd


def show_argparse(args):
    args_dict = vars(args)
    df = pd.DataFrame({'argument': args_dict.keys(), 'value': args_dict.values()})
    print('*' * 15)
    print(df)
    print('*' * 15)
