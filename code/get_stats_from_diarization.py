import argparse
import json
import numpy as np
import os

parser = argparse.ArgumentParser("get diar stats")
parser.add_argument("diar_file")

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.diar_file, "r") as f:
        diarization = json.load(f)

    maxspk = 3
    stats = [0]*maxspk
    for sess in diarization.keys():
        maxlen = max([diarization[sess][spk][-1][-1] for spk in diarization[sess].keys()])

        dummy = np.zeros(int(np.ceil(maxlen/160)), dtype="uint8")

        for spk in diarization[sess].keys():
            if spk == "garbage":
                continue
            for s, e in diarization[sess][spk]:
                s = int(np.floor(s/160))
                e = int(np.floor(e/160))
                dummy[s:e] += 1 

        for i in range(len(stats)):
            stats[i] += len(np.where(dummy == i)[0])
        #assert not np.where(dummy > maxspk)[0].any()

    for i in range(len(stats)):
        print("TOT {} SPK {}".format(i, stats[i]))






