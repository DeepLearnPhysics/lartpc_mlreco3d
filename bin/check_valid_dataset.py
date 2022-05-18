# Script to mark all bad ROOT files before merging them with hadd
# ================================================================
#
# Usage: python3 bin/check_valid_dataset.py bad_files.txt file1.root file2.root ... fileN.root
#
# Output: will write a list of bad files in bad_files.txt
# (one per line) that can then be used to move or remove
# these bad files before doing hadd. For example using:
#
# $ for file in $(cat bad_files.txt); do mv "$file" bad_files/; done
#
# What it does:
# Loop over all TTrees in a given ROOT file and check that
# they have the same number of entries.
#
from ROOT import TCanvas, TPad, TFile, TPaveLabel, TPaveText, TChain
from ROOT import gROOT
import ROOT
import pandas as pd
import numpy as np
import argparse


if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Check validity of dataset")
    argparse.add_argument("output_file", type=str, help="output text file to write bad files names")
    argparse.add_argument("files", type=str, nargs="+", help="files to check")

    args = argparse.parse_args()

    # print(args)

    output = open(args.output_file, 'w')
    bad_files = []
    global_keys = []
    counts = []

    def mark_bad_file(file):
        output.write(file + '\n')
        bad_files.append(file)

    for idx, file in enumerate(args.files):
        print(file)
        f = TFile(file)
        keys = [key.GetName() for key in f.GetListOfKeys()]
        global_keys.append(keys)

        # If keys is a subset of global_keys or global_keys is shorter
        # if global_keys is None:
        #     global_keys = keys
        # elif len(np.intersect1d(keys, global_keys)) < len(global_keys):
        #     # keys is a subset of global keys
        #     mark_bad_file(file)
        #     continue
        # elif len(np.intersect1d(keys, global_keys)) < len(keys):
        #     # global_keys is a subset of keys
        #     if arg.files[idx-1] not in bad_files:
        #         mark_bad_file(arg.files[idx-1])
        #         global_keys = keys
                # note that's assuming we don't get 2 files in a row with bad keys...

        # print(keys)

        trees = [f.Get(key) for key in keys]

        nentries = [tree.GetEntries() for tree in trees]
        counts.append(len(np.unique(nentries)))
        # print(nentries)

        # if len(np.unique(nentries)) != 1:
        #     mark_bad_file(file)

    all_keys = np.unique(np.hstack(global_keys))
    #print(all_keys)
    # Function testing equality of two lists of strings
    def is_equal(a, b):
        c = np.intersect1d(a, b)
        return len(c) == len(a) and len(c) == len(b)

    for idx, file in enumerate(args.files):
        if counts[idx] != 1 or not is_equal(np.unique(global_keys[idx]), all_keys):
            mark_bad_file(file)
            # print(len(global_keys[idx]), len(all_keys))
            # print(counts[idx], is_equal(global_keys[idx], all_keys))

    print('\nFound bad files: ')
    for f in bad_files:
        print(f)

    output.close()
