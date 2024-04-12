import argparse
import os
import glob
import json
from pathlib import Path
import re

parser = argparse.ArgumentParser("Parsing Forced alignments")
parser.add_argument("falign_dir")
parser.add_argument("output_json")

def parse_falign_file(falign_file):

    diarization = {}
    with open(falign_file, "r") as f:
        for line in f:
            l = line.split(" ")
            start = int(float(l[3])*16000)
            stop = int((float(l[3])+float(l[4]))*16000)
            spk = re.findall("(speaker[0-9]+)", l[-3])[0]
            if spk not in diarization.keys():
                diarization[spk] = [[start, stop]]
            else:
                diarization[spk].append([start, stop])


    return diarization

if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(Path(args.output_json).parent, exist_ok=True)
    falign_sessions = glob.glob(os.path.join(args.falign_dir, "*.rttm"))
    assert len(falign_sessions) > 0, "empty folder ?"

    diarization = {}
    for f in falign_sessions:
        sess = Path(f).stem
        diarization[sess] = parse_falign_file(f)


    with open(args.output_json, "w") as f:
        json.dump(diarization, f, indent=4)
