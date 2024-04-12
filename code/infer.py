import torch
import os
import argparse
from glob import glob
import soundfile as sf
from torchaudio.compliance.kaldi import mfcc
from osdc.utils.oladd import overlap_add
import numpy as np
from osdc.features.ola_feats import compute_feats_windowed
import yaml
from train import OSDC_AMI

parser = argparse.ArgumentParser("Single-Channel inference, average logits")
parser.add_argument("--exp_dir", type=str, default="../exp/tcn")
parser.add_argument("--checkpoint_name", type=str, default="../checkpoints-v3.ckpt")
parser.add_argument("--wav_dir", type=str, default="/scratch/users/ntu/alibabaz/amicorpus")
parser.add_argument("--out_dir", type=str, default="../exp/tcn/preds/checkpoints-v3.ckpt")
parser.add_argument("--gpus", type=str, default="0")
parser.add_argument("--window_size", type=int, default=200)
parser.add_argument("--lookahead", type=int, default=200)
parser.add_argument("--lookbehind", type=int, default=200)

def plain_single_file_predict(model, wav_dir, train_configs, out_dir, window_size=400, lookahead=200, lookbehind=200):

    model = model.eval().cuda()
    wavs = glob(os.path.join(wav_dir, "*.flac"), recursive=True)

    assert len(wavs) > 0, "No file found"

    for wav in wavs:
        print("Processing File {}".format(wav))
        audio, _ = sf.read(wav)


        if train_configs["feats"]["type"] == "mfcc_kaldi":
            feats_func = lambda x: mfcc(torch.from_numpy(x.astype("float32").reshape(1, -1)),
                                             **train_configs["mfcc_kaldi"]).transpose(0, 1)
        else:
            raise NotImplementedError

        tot_feats = compute_feats_windowed(feats_func, audio)
        tot_feats = tot_feats.detach().cpu().numpy()
        print(tot_feats.shape)
        pred_func = lambda x : model(torch.from_numpy(x).unsqueeze(0).cuda()).detach().cpu().numpy()
        preds = overlap_add(tot_feats, pred_func, window_size, window_size // 2, lookahead=lookahead, lookbehind=lookbehind)
        out_file = os.path.join(out_dir, wav.split("/")[-1].split(".wav")[0] + ".logits")
        np.save(out_file, preds)


if __name__ == "__main__":
    args = parser.parse_args()
    with open(os.path.join(args.exp_dir, "confs.yml"), "r") as f:
        confs = yaml.load(f, Loader=yaml.FullLoader)
    # test if compatible with lightning
    confs.update(args.__dict__)

    model = OSDC_AMI(confs)
    if confs["checkpoint_name"].startswith("avg"):
        state_dict = torch.load(os.path.join(confs["exp_dir"], confs["checkpoint_name"]),
                                map_location='cpu')

    else:

        state_dict = torch.load(os.path.join(confs["exp_dir"], confs["checkpoint_name"]),
                            map_location='cpu')["state_dict"]

    model.load_state_dict(state_dict)
    model = model.model
    os.makedirs(confs["out_dir"], exist_ok=True)
    plain_single_file_predict(model, confs["wav_dir"],
                              confs, confs["out_dir"], window_size=args.window_size,
                              lookahead=args.lookahead, lookbehind=args.lookbehind)
