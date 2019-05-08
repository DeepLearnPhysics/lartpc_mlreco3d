#!/usr/bin/python
import os
import sys
import yaml

current_directory = os.path.dirname(os.path.abspath(__file__))
current_directory = os.path.dirname(current_directory)
sys.path.insert(0, current_directory)
import mlreco
from mlreco.flags import Flags


def main():
    cfg_file = sys.argv[1]
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(current_directory, 'config', sys.argv[1])
    if not os.path.isfile(cfg_file):
        print(sys.argv[1], 'not found...')
        sys.exit(1)

    cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)

    flags = Flags(cfg)
    flags.parse_args()


if __name__ == '__main__':
    main()
