# Script to count events in a dataset
# ===================================
#
# Usage: python3 bin/count_events_in_dataset.py sparse3d_reco_cryoE file1.rot file2.root ... fileN.root
#
# Output: will write in stdout the total event count.

from ROOT import TCanvas, TPad, TFile, TPaveLabel, TPaveText, TChain
from ROOT import gROOT
import ROOT
import pandas as pd
import numpy as np
import argparse

if __name__ == "__main__":
    argparse = argparse.ArgumentParser(description="Count events in dataset")
    argparse.add_argument("key", type=str, help="TTree name to use to count events")
    argparse.add_argument("files", type=str, nargs="+", help="files to check")

    args = argparse.parse_args()
    entries_count = []
    for idx, file in enumerate(args.files):
        f = TFile(file)
        tree = f.Get("%s_tree" % args.key)
        nentries = tree.GetEntries()
        entries_count.append(nentries)
        f.Close()
        print("%s... done. (%d events)" % (file, nentries))

    print("Counted %d events" % np.sum(entries_count))
