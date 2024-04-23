from torch.utils.data import Dataset
import glob
import os
from pathlib import Path
import soundfile as sf
import numpy as np
import torch
from osdc.utils.oladd import _gen_frame_indices
import random
from pysndfx import AudioEffectsChain

class OnlineFeats(Dataset):

    def __init__(self, ami_audio_root, label_root, configs, segment=300, probs=None, synth=None):

        self.configs = configs
        self.segment = segment
        self.probs = probs
        self.synth = None

        # audio_files： list  存储音频文件名
        # labels： 文件列表，存储标签文件名
        # lab_hash：sess to (json file)
        # devices：判断是否有label，并存储音频文件列表
        # devices_hash：sess to (wav file)


        # 读取每一个文件的音频和标签
        audio_files = glob.glob(os.path.join(ami_audio_root, "*.flac"), recursive=True)
        for f in audio_files:
            if len(sf.SoundFile(f)) < self.segment:
                print("Dropping file {}".format(f))
        labels = glob.glob(os.path.join(label_root, "*.wav"))
        lab_hash = {}

        # 处理标签:::  使用hash表存储标签  LABEL-XXXX.wav => XXXX
        for l in labels:
            l_sess = str(Path(l).stem).split("-")[-1]
            lab_hash[l_sess] = l

        self.label_hash = lab_hash


        # 处理音频文件
        devices_hash = {}
        devices = []
        for f in audio_files:
            sess = Path(f).stem
            if sess not in lab_hash.keys():
                print("Skip session because we have no labels for it")
                continue
            devices.append(f)   # 保存音频文件
            if sess not in devices_hash.keys():
                devices_hash[sess] = [f]
            else:
                devices_hash[sess].append(f)

        self.devices = devices
        self.devices_hash = devices_hash # used for data augmentation

        #assert len(set(list(meta.keys())).difference(set(list(lab_hash.keys())))) == 0
        # remove keys

        

        if self.probs: # parse for data-augmentation
            label_one = []
            label_two = []

            for l in labels:
                c_label, _ = sf.read(l)  # read it all
                sess = Path(l).stem.split("-")[-1]
                # find contiguous
                tmp = self.get_segs(c_label, 1, 1)   # 返回一个列表，列表中的元素是一个区间
                for s,e in tmp:
                    # 验证，不能存在大于1的标签
                    assert not np.where(c_label[s:e] > 1)[0].any()
                tmp = [(sess, x[0], x[1]) for x in tmp]  # we need session also
                label_one.extend(tmp)

                # do the same for two speakers
                tmp = self.get_segs(c_label, 2, 2)
                for s, e in tmp:
                    # 验证，不能存在不等于2的标签
                    assert not np.where(c_label[s:e] != 2)[0].any()
                tmp = [(sess, x[0], x[1]) for x in tmp]
                label_two.extend(tmp)

            self.label_one = label_one
            self.label_two = label_two

        # 所有语音的长度 / 段数  =  每段长 
        self.tot_length = int(np.sum([len(sf.SoundFile(l)) for l in labels]) / segment)

        self.set_feats_func()

        if synth:
            self.synth=synth
            # using synthetic data.

    def get_segs(self, label_vector, min_speakers,  max_speakers):

        segs = []
        # 寻找哪些位置的label_vector的值在min_speakers和max_speakers之间
        label_vector =  np.logical_and(label_vector <= max_speakers, label_vector >= min_speakers)
        # label_vector是0/1串，通过前后比较获取拿下位置的值进行了改变  ==>   提取连续块
        # 该端点值为还未改变   0,0,0,0,1,1,0
        #                           |

        # ================================== 原代码这里存在问题 ==========================================
        # ========================== 由于没有搞清楚开闭区间，导致结构混乱 =================================
        # 这里生成的应该是闭区间，但是其以开区间处理，导致最后一个区间的右端点没有被加入
        changePoints = np.where((label_vector[:-1] != label_vector[1:]) == True)[0] + 1
        # changePoints存储每一段的左端点（闭区间）
        changePoints = np.concatenate((np.array(0).reshape(1, ), changePoints))
        
        # 确保第一个区间是 1 区间，其要提取1区间
        if label_vector[0] == 1:    start = 0
        else:       start = 1

        # 仅仅提取1区间，因此步长为2
        for i in range(start, len(changePoints) - 1, 2):
            if (changePoints[i + 1] - changePoints[i]) > 30: # if only more than 30 frames
                segs.append([changePoints[i], changePoints[i + 1]-1])
        # ===============================================================================================

        return segs

    def mfcc_kaldi(self, x):
        from torchaudio.compliance.kaldi import mfcc
        return mfcc(torch.from_numpy(x.astype("float32").reshape(1, -1)), **self.configs["mfcc_kaldi"]).transpose(0, 1)
    def fbank_kaldi(self, x):
        from torchaudio.compliance.kaldi import fbank
        return fbank(torch.from_numpy(x.astype("float32").reshape(1, -1)), **self.configs["fbank_kaldi"]).transpose(0, 1)
    def spectrogram_kaldi(self, x):
        from torchaudio.compliance.kaldi import spectrogram
        return spectrogram(torch.from_numpy(x.astype("float32").reshape(1, -1)), **self.configs["spectrogram_kaldi"]).transpose(0, 1)

    def set_feats_func(self):

        # initialize feats_function
        if self.configs["feats"]["type"] == "mfcc_kaldi":
            self.feats_func = self.mfcc_kaldi
        elif self.configs["feats"]["type"] == "fbank_kaldi":
            self.feats_func = self.fbank_kaldi
        elif self.configs["feats"]["type"] == "spectrogram_kaldi":
            self.feats_func = self.spectrogram_kaldi
        elif self.configs["feats"]["type"] == "orig":
            self.feats_func = lambda x: x
        else:
            raise NotImplementedError

    def __len__(self):
        return self.tot_length

    def noaugm(self):
        # no augmentation
        file = np.random.choice(self.devices)
        sess = Path(file).stem.split(".")[0]

        # 读取该文件，并截取segment长度的音频
        start = np.random.randint(1, len(sf.SoundFile(self.label_hash[sess])) - self.segment - 2)
        stop = start + self.segment
        label, _ = sf.read(self.label_hash[sess], start=start, stop=stop)
        if self.configs["task"] == "vad":
            label = label >= 1
        elif self.configs["task"] == "osd":
            label = label >= 2
        elif self.configs["task"] == "vadosd":
            label = np.clip(label, 0, 2)
        elif self.configs["task"] == "count":
            pass
        else:
            raise EnvironmentError
        
        frame_size = self.configs["data"]["fs"] * self.configs["feats"]["frame_size"]
        hop_size = self.configs["data"]["fs"] * self.configs["feats"]["hop_size"]
        # get file start
        start = int((start - 1) * hop_size)
        stop = int((stop - 2) * hop_size + frame_size)

        audio, fs = sf.read(file, start=start, stop=stop)
       
        if len(audio.shape) > 1:  # binaural
            audio = audio[:, np.random.randint(0, 1)]

        # audio = self.feats_func(audio)
        # label = label[:audio.shape[-1]]


        return audio, torch.from_numpy(label).long(), torch.ones(len(label)).bool()

    @staticmethod
    def normalize(signal, target_dB):

        fx = (AudioEffectsChain().custom(
            "norm {}".format(target_dB)))
        signal = fx(signal)
        return signal

    def __getitem__(self, item):

        if not self.probs:
            return self.noaugm()
        else:
            spkrs = np.random.choice([1, 4], p=self.probs)
            if spkrs == 1:
                return self.noaugm()
            elif spkrs ==4:
                # sample 2 from labels one
                mix = []
                labels = []
                first_lvl = None
                maxlength = None
                for i in range(spkrs):
                    sess, start, stop = random.choice(self.label_one)
                    label, _ = sf.read(self.label_hash[sess], start=start, stop=stop)

                    frame_size = self.configs["data"]["fs"] * self.configs["feats"]["frame_size"]
                    hop_size = self.configs["data"]["fs"] * self.configs["feats"]["hop_size"]

                    # get file start
                    start = int((start - 1) * hop_size)
                    stop = int((stop - 1) * hop_size + frame_size)
                    file = np.random.choice(self.devices_hash[sess])
                    audio, fs = sf.read(file, start=start, stop=stop)
                    if len(audio.shape) > 1:  # binaural
                        audio = audio[:, np.random.randint(0, 1)]
                        
                    if i == 0:
                        c_lvl = np.clip(random.normalvariate(*self.configs["augmentation"]["abs_stats"]), -30, -4) # allow for clipping  in CHiME some devices are clipped
                        first_lvl = c_lvl
                        audio = self.normalize(audio, c_lvl)
                        maxlength = len(audio)
                    else:
                        c_lvl = np.clip(first_lvl - random.normalvariate(*self.configs["augmentation"]["rel_stats"]), first_lvl-10, min(first_lvl+10, -4))
                        audio = self.normalize(audio, c_lvl)
                        rand_offset = random.randint(0, maxlength)
                        # pad only heads
                        audio = np.pad(audio, (rand_offset, 0), 'constant')
                        n_label = int(np.floor((len(audio) - frame_size) / hop_size) + 1)
                        label = np.pad(label, (n_label - len(label), 0), 'constant')
                        maxlength = max(len(audio), maxlength)

                    mix.append(audio)
                    labels.append(label)

                assert maxlength == max([len(x) for x in mix])
                seg_len = int((self.segment - 1) * hop_size + frame_size)
                if maxlength > seg_len:
                    mix = [x[:seg_len] for x in mix]
                    labels = [x[:self.segment] for x in labels]
                    valid = torch.ones(self.segment).bool()
                else:
                    valid = torch.ones(self.segment).bool()
                    label_len = int((maxlength - frame_size) // hop_size + 1)
                    valid[label_len:] = False

                # 混合音频
                mix = [np.pad(x, (0, seg_len - len(x)), 'constant') for x in mix]
                mix = np.sum(np.stack(mix), 0)
                mix = np.clip(mix, -1, 1)  # clipping audio

                padlen = self.segment
                labels = [np.pad(x, (0, padlen - len(x)), 'constant') for x in labels]
                labels = np.sum(np.stack(labels), 0)
                mix = self.feats_func(mix)
                labels = labels[:mix.shape[-1]]
                if self.configs["task"] == "vadosd":
                    labels = np.clip(labels, 0, 2)
                valid = valid[:mix.shape[-1]]
                return mix, torch.from_numpy(labels).long(), valid

            elif spkrs == 3:
                pass
                # sample 1 from label one and two from label two
            elif spkrs == 4:
                pass
                # sample 2 from label one and two from label two

# class OnlineChunkedFeats(Dataset):

#     def __init__(self, chime6_root, split, label_root, configs, segment=300):

#         self.configs = configs
#         self.segment = segment
#         meta = parse_chime6(chime6_root, split)

#         devices = {}
#         for sess in meta.keys():
#             devices[sess] = []
#             for array in meta[sess]["arrays"].keys():
#                 devices[sess].extend(meta[sess]["arrays"][array]) # only channel 1

#         labels = glob.glob(os.path.join(label_root, "*.wav"))
#         lab_hash = {}

#         for l in labels:
#             l_sess = str(Path(l).stem).split("-")[-1]
#             lab_hash[l_sess] = l

#         self.lab_hash = lab_hash
#         chunks = self.get_chunks(labels)

#         examples = []
#         for sess in chunks.keys():
#             for s, e in chunks[sess]:
#                 for dev in devices[sess]:
#                     examples.append((dev, s, e))

#         self.examples = examples

#         self.set_feats_func()

#     def set_feats_func(self):

#         # initialize feats_function
#         if self.configs["feats"]["type"] == "mfcc_kaldi":
#             from torchaudio.compliance.kaldi import mfcc
#             self.feats_func = lambda x: mfcc(torch.from_numpy(x.astype("float32").reshape(1, -1)),
#                                              **self.configs["mfcc_kaldi"]).transpose(0, 1)
#         elif self.configs["feats"]["type"] == "fbank_kaldi":
#             from torchaudio.compliance.kaldi import fbank
#             self.feats_func = lambda x: fbank(torch.from_numpy(x.astype("float32").reshape(1, -1)),
#                                               **self.configs["fbank_kaldi"]).transpose(0, 1)
#         elif self.configs["feats"]["type"] == "spectrogram_kaldi":
#             from torchaudio.compliance.kaldi import spectrogram
#             self.feats_func = lambda x: spectrogram(torch.from_numpy(x.astype("float32").reshape(1, -1)),
#                                                     **self.configs["spectrogram_kaldi"]).transpose(0, 1)
#         else:
#             raise NotImplementedError

#     def get_chunks(self, labels):

#         chunks = {}
#         chunk_size = self.configs["data"]["segment"]
#         frame_shift = self.configs["data"]["segment"]

#         for l in labels:
#             sess = Path(l).stem.split("-")[-1]
#             chunks[sess] = []
#             # generate chunks for this file
#             c_length = len(sf.SoundFile(l)) # get the length of the session files in samples
#             for st, ed in _gen_frame_indices(
#                     c_length, chunk_size, frame_shift, use_last_samples=False):
#                 chunks[sess].append([st, ed])
#         return chunks


#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, item):

#         device, s, e = self.examples[item]
#         sess = get_session(device)
#         labelfile = self.lab_hash[sess]

#         label, _ = sf.read(labelfile, start=s, stop=e)
#         if self.configs["task"] == "vad":
#             label = label >= 1
#         elif self.configs["task"] == "osd":
#             label = label >= 2
#         elif self.configs["task"] == "vadosd":
#             label = np.clip(label, 0, 2)
#         elif self.configs["task"] == "count":
#             pass
#         else:
#             raise EnvironmentError

#         start = int(s * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"])
#         stop = int(e * self.configs["data"]["fs"] * self.configs["feats"]["hop_size"] +
#                    self.configs["data"]["fs"] * self.configs["feats"]["hop_size"] * 2)

#         audio, fs = sf.read(device, start=start, stop=stop)

#         if len(audio.shape) > 1:  # binaural
#             audio = audio[:, np.random.randint(0, 1)]

#         audio = self.feats_func(audio)
#         assert audio.shape[-1] == len(label)
#         return audio, torch.from_numpy(label).long()


if __name__ == "__main__":

    import yaml
    with open(r"/home/getsum/code/speech/DIHARD3/conf/fine_tune.yml", "r") as f:
        confs = yaml.load(f, Loader=yaml.FullLoader)


    # a = OnlineChunkedFeats("/media/sam/bx500/amicorpus/audio/", "/home/sam/Desktop/amicorpus/labels/train/", confs)

    a = OnlineFeats(r"/home/getsum/data/DIHARD3/third_dihard_challenge_dev/data/flac",
                   r"/home/getsum/data/DIHARD3/fa_labels/dev", confs, segment=500, probs=[0.3, 0.7])
    from torch.utils.data import DataLoader
    for x,y,z in DataLoader(a, batch_size=8, shuffle=True):
        print(x.shape, y.shape, z.shape)
        break
        




