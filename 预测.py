from __future__ import division

import numpy as np


def get_M(self, p, q):  # 获取对应扩散率曲线的最优潜在市场容量
    diffu = Diffuse(p, q, self.s_len, G=self.G)
    s_estim = diffu.repete_diffuse()
    x = np.mean(s_estim, axis=0)
    a = np.sum(np.square(x)) / np.sum(self.s)  # 除以np.sum(self.s)是为减少a的大小
    b = -2 * np.sum(x * self.s) / np.sum(self.s)
    c = np.sum(np.square(self.s)) / np.sum(self.s)
    mse, sigma = np.sqrt(sum(self.s) * (4 * a * c - b ** 2) / (4 * a * self.s_len)), -b / (2 * a)
    m = sigma * self.G.number_of_nodes()
    return mse, p, q, m, x * sigma