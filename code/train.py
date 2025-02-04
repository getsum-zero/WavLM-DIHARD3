import os
import yaml
import argparse
from torch import nn
import torch
from collections import OrderedDict
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from osdc.utils import BinaryMeter, MultiMeter
from online_data import OnlineFeats

parser = argparse.ArgumentParser(description="OSDC on DIHARD3")
parser.add_argument("--conf_file", type=str, default="conf/fine_tune.yml")
parser.add_argument("--log_dir", type=str, default="exp/wavlm")
parser.add_argument("--gpus", type=str, default="0")


class PlainModel(nn.Module):

    def __init__(self, masker, ckpt = None):

        super(PlainModel, self).__init__()
        self.model = masker
        if ckpt is not None:
            self.model.load_state_dict(ckpt["model"])
        self.normalize = ckpt["cfg"]["normalize"]

    def forward(self, tf_rep):
        if self.normalize:
            tf_rep = torch.nn.functional.layer_norm(tf_rep , tf_rep.shape)
        
        mask, _ = self.model.extract_features(tf_rep)
        return mask


class OSDC_DIHARD3(pl.LightningModule):


    def __init__(self, hparams):
        super(OSDC_DIHARD3, self).__init__()
        self.configs = hparams # avoid pytorch-lightning hparams logging

        # 样本不平衡：输入每一类的权重
        # ====== ===========================
        cross = nn.CrossEntropyLoss((1/torch.Tensor(self.configs["augmentation"]["probs"])).cuda(), reduction="none")


        self.loss = lambda x, y : cross(x, y)  #+ 0.1*dice(1-x, 1-y) # flip positive for focal loss
        self.train_count_metrics = MultiMeter()
        self.train_vad_metrics = BinaryMeter()
        self.train_osd_metrics = BinaryMeter()
        self.val_count_metrics = MultiMeter()
        self.val_vad_metrics = BinaryMeter()
        self.val_osd_metrics = BinaryMeter()

        from osdc.models.WavLM import WavLM, WavLMConfig
        checkpoint = torch.load(confs["training"]["resume_from"])
        cfg = WavLMConfig(checkpoint['cfg'])
        self.model = PlainModel(WavLM(cfg), checkpoint)



    def forward(self, *args, **kwargs):
        pass

    def training_step(self, batch, batch_idx):

        feats, label, mask = batch
        preds = self.model(feats)
        loss = self.loss(preds, label)
        loss = loss*mask.detach()
        loss = loss.mean()
        preds = torch.softmax(preds, 1)
        self.train_count_metrics.update(torch.argmax(preds, 1), label)
        self.train_vad_metrics.update(torch.sum(preds[:, 1:], 1), label >= 1)
        self.train_osd_metrics.update(torch.sum(preds[:, 2:], 1), label >= 2)


        tensorboard_logs = {'train_batch_loss': loss,
                            'train_tp_count': self.train_count_metrics.get_tp(),
                            'train_tn_count': self.train_count_metrics.get_tn(),
                            'train_fp_count': self.train_count_metrics.get_fp(),
                            'train_fn_count': self.train_count_metrics.get_fn(),
                            'train_prec_count': self.train_count_metrics.get_precision(),
                            'train_rec_count': self.train_count_metrics.get_recall(),
                            'train_prec_vad': self.train_vad_metrics.get_precision(),
                            'train_rec_vad': self.train_vad_metrics.get_recall(),
                            'train_fa_vad': self.train_vad_metrics.get_fa(),
                            'train_miss_vad': self.train_vad_metrics.get_miss(),
                            'train_der_vad': self.train_vad_metrics.get_der(),
                            'train_prec_osd': self.train_osd_metrics.get_precision(),
                            'train_rec_osd': self.train_osd_metrics.get_recall(),
                            'train_fa_osd': self.train_osd_metrics.get_fa(),
                            'train_miss_osd': self.train_osd_metrics.get_miss(),
                            'train_der_osd': self.train_osd_metrics.get_der(),
                            'train_tot_silence': self.train_count_metrics.get_positive_examples_class(0),
                            'train_tot_1spk': self.train_count_metrics.get_positive_examples_class(1),
                            'train_tot_2spk': self.train_count_metrics.get_positive_examples_class(2),
                            'train_tot_3spk': self.train_count_metrics.get_positive_examples_class(3),
                            'train_tot_4spk': self.train_count_metrics.get_positive_examples_class(4)
                            }

        output = OrderedDict({
                'loss': loss,
                'log': tensorboard_logs
            })
        return output

    '''
    # def validation_step(self, batch, batch_indx):

    #     feats, label, _ = batch
    #     preds = self.model(feats)
    #     loss = self.loss(preds, label).mean()
    #     preds = torch.softmax(preds, 1)
    #     self.val_count_metrics.update(torch.argmax(preds, 1), label)
    #     self.val_vad_metrics.update(torch.sum(preds[:, 1:], 1), label >= 1)
    #     #self.val_osd_metrics.update(torch.argmax(torch.cat((preds[:, :2], torch.sum(preds[:, 2:], 1, keepdim=True)),1),1), torch.clamp(label, 0, 2))
    #     self.val_osd_metrics.update(torch.sum(preds[:, 2:], 1), label >= 2)
    #     tqdm_dict = {'val_loss': loss}

    #     output = OrderedDict({
    #         'val_loss': loss,
    #         'progress_bar': tqdm_dict,
    #     })

    #     return output

    # def validation_step_end(self, outputs):

    #     avg_loss = outputs["val_loss"].mean()
    #     tqdm_dict = {'val_loss': avg_loss}
    #     tensorboard_logs = {'val_loss': avg_loss,
    #                         'val_tp_count': self.val_count_metrics.get_tp(),
    #                         'val_tn_count': self.val_count_metrics.get_tn(),
    #                         'val_fp_count': self.val_count_metrics.get_fp(),
    #                         'val_fn_count': self.val_count_metrics.get_fn(),
    #                         'val_prec_count': self.val_count_metrics.get_precision(),
    #                         'val_rec_count': self.val_count_metrics.get_recall(),
    #                         'val_prec_vad': self.val_vad_metrics.get_precision(),
    #                         'val_rec_vad': self.val_vad_metrics.get_recall(),
    #                         'val_fa_vad': self.val_vad_metrics.get_fa(),
    #                         'val_miss_vad': self.val_vad_metrics.get_miss(),
    #                         'val_der_vad': self.val_vad_metrics.get_der(),
    #                         'val_prec_osd': self.val_osd_metrics.get_precision(),
    #                         'val_rec_osd': self.val_osd_metrics.get_recall(),
    #                         'val_fa_osd': self.val_osd_metrics.get_fa(),
    #                         'val_miss_osd': self.val_osd_metrics.get_miss(),
    #                         'val_der_osd': self.val_osd_metrics.get_der(),
    #                         }

    #     self.train_count_metrics.reset()
    #     self.train_vad_metrics.reset()
    #     self.train_osd_metrics.reset()
    #     self.val_count_metrics.reset()
    #     self.val_vad_metrics.reset()
    #     self.val_osd_metrics.reset()
    #     output = OrderedDict({
    #         'val_loss': avg_loss,
    #         'progress_bar': tqdm_dict,
    #         'log': tensorboard_logs
    #     })

    #     return output
    '''

    def configure_optimizers(self):

        opt = torch.optim.Adam(self.model.parameters(),
                                    self.configs["opt"]["lr"], weight_decay=self.configs["opt"]["weight_decay"])
        # ReduceLROnPlateau 当性能不再增加时，减小学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt)

        return {
            'optimizer': opt,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


    def train_dataloader(self):
        dataset = OnlineFeats(self.configs["data"]["data_root_train"], self.configs["data"]["label_train"],
                              self.configs, probs=self.configs["augmentation"]["probs"], segment=self.configs["data"]["segment"])
        
        dataloader = DataLoader(dataset, batch_size=self.configs["training"]["batch_size"],
                                shuffle=True, num_workers=self.configs["training"]["num_workers"], drop_last=True)
        return dataloader


    # def val_dataloader(self):

    #     dataset = OnlineFeats(self.configs["data"]["data_root_val"], self.configs["data"]["label_val"],
    #                                 self.configs, segment=self.configs["data"]["segment"])
    #     dataloader = DataLoader(dataset, batch_size=self.configs["training"]["batch_size"],
    #                             shuffle=True, num_workers=self.configs["training"]["num_workers"], drop_last=True)

    #     return dataloader

if __name__ == "__main__":

    args = parser.parse_args()
    with open(args.conf_file, "r") as f:
        confs = yaml.load(f, Loader=yaml.FullLoader)

    # test if compatible with lightning
    confs.update(args.__dict__)
    net = OSDC_DIHARD3(confs)

    checkpoint_dir = os.path.join(confs["log_dir"], 'checkpoints/')
    checkpoint = ModelCheckpoint(checkpoint_dir, monitor='val_loss',
                                 mode='min',  verbose=True, save_top_k=5)

    # 当到达性能不在改变时，停止训练
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )

    # 存储配置文件
    with open(os.path.join(confs["log_dir"], "confs.yml"), "w") as f:
        yaml.dump(confs, f)

    # 以log_dir目录名存储log
    logger = TensorBoardLogger(os.path.dirname(confs["log_dir"]), confs["log_dir"].split("/")[-1])

    trainer = pl.Trainer(max_epochs=confs["training"]["n_epochs"], # gpus=confs["gpus"],
                         accumulate_grad_batches=confs["training"]["accumulate_batches"], callbacks=[checkpoint, early_stop_callback],
                         logger = logger,
                         gradient_clip_val=confs["training"]["gradient_clip"],
                         accelerator = "gpu" if torch.cuda.is_available() else "cpu",
                         strategy = "auto",
                         devices = "auto",
                    )
    trainer.fit(net)