"""_trainTestValidSplit_
Coder: LeninGF
Date: 2022-12-29

Target: to organize dataset in 3 csv files: train, valid and test.
These files will be used later by HuggingFace Dataset library to create
the corresponding datasets for training, validating and testing

Task:
- where to read data from
- where to write data to
- define the amounts for train, valid and test with json?
"""

import pandas as pd
import numpy as np
import os
from argparse import ArgumentParser

def main():
    print("hello world")
    return 0
    
if __name__ == "__main__":
    parser = ArgumentParser(description='Script to split dataset in train valid and test datasets')
    parser.add_argument()
    main()
