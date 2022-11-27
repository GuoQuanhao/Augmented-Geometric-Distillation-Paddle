# -*- coding: utf-8 -*-
# Time    : 2021/8/6 16:37
# Author  : Yichen Lu
import paddle


def cos(x, y):
    x_ = x / x.norm(axis=1, keepdim=True)
    y_ = y / y.norm(axis=1, keepdim=True)
    return paddle.mm(x_, y_.t())


def euclidean_dist(x, y, *args):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.shape[0], y.shape[0]
    xx = paddle.pow(x, 2).sum(1, keepdim=True).expand([m, n])
    yy = paddle.pow(y, 2).sum(1, keepdim=True).expand([n, m]).t()
    dist = xx + yy
    dist = paddle.addmm(input=dist, x=x, y=y.t(), beta=1, alpha=-2)
    dist = dist.clip(min=1e-12).sqrt()  # for numerical stability
    return dist