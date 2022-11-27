from __future__ import print_function, absolute_import
import os.path as osp

import paddle
from .osutils import mkdir_if_missing


def save_checkpoint(state, fpath='checkpoint.pdparams'):
    mkdir_if_missing(osp.dirname(fpath))
    paddle.save(state, fpath)


def load_checkpoint(fpath):
    assert osp.isdir(fpath) or osp.isfile(fpath), 'previous checkpoint path not exists or not a folder'
    fpath = osp.join(fpath, 'checkpoint.pdparams') if osp.isdir(fpath) else fpath

    if osp.isfile(fpath):
        checkpoint = paddle.load(fpath)
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


class CheckpointManager(object):
    def __init__(self, logs_dir='./logs', **modules):
        self.logs_dir = logs_dir
        self.modules = modules

    def save(self, epoch, fpath=None, **modules):
        ckpt = {}
        modules.update(self.modules)
        for name, module in modules.items():
            if isinstance(module, paddle.DataParallel):
                ckpt[name] = module.module.state_dict()
            else:
                ckpt[name] = module.state_dict()
        ckpt['epoch'] = epoch + 1

        fpath = osp.join(self.logs_dir, f"checkpoint-epoch{epoch}.pdparams") if fpath is None else fpath
        save_checkpoint(ckpt, fpath)

    def load(self, ckpt):
        for name, module in self.modules.items():
            module.set_state_dict(ckpt.get(name, {}))
            print(f"Loading {name}... \n")
        return ckpt["epoch"]
