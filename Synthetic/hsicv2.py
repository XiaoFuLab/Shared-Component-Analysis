"""
python implementation of Hilbert Schmidt Independence Criterion
hsic_gam implements the HSIC test using a Gamma approximation
Python 2.7.12

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., 
& Smola, A. J. (2007). A kernel statistical test of independence. 
In Advances in neural information processing systems (pp. 585-592).

Shoubo (shoubo.sub AT gmail.com)
09/11/2016

Inputs:
X 		n by dim_x matrix
Y 		n by dim_y matrix
alph 		level of test

Outputs:
testStat	test statistics
thresh		test threshold for level alpha test
"""

from __future__ import division
import numpy as np
from scipy.stats import gamma

import torch

def rbf_dot_torch(pattern1, pattern2, deg):
    size1 = pattern1.shape
    size2 = pattern2.shape

    G = torch.sum(pattern1*pattern1, 1).reshape(size1[0],1)
    H = torch.sum(pattern2*pattern2, 1).reshape(size2[0],1)

    Q = G.repeat(1, size2[0])
    R = H.t().repeat(size1[0], 1)

    H = Q + R - 2* torch.mm(pattern1, pattern2.t())

    H = torch.exp(-H/2/(deg**2))

    return H

def hsic_gam_torch(X, Y, alph = 0.5):
    n = X.shape[0]

    Xmed = X

    G = torch.sum(Xmed*Xmed, 1).reshape(n,1)
    Q = G.repeat(1, n)
    R = G.t().repeat(n, 1)

    dists = Q + R - 2* torch.mm(Xmed, Xmed.t())
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_x = torch.sqrt( 0.5 * torch.median(dists[dists>0]) )

    Ymed = Y

    G = torch.sum(Ymed*Ymed, 1).reshape(n,1)
    Q = G.repeat(1, n)
    R = G.t().repeat(n, 1)

    dists = Q + R - 2* torch.mm(Ymed, Ymed.t())
    dists = dists - torch.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = torch.sqrt( 0.5 * torch.median(dists[dists>0]) )

    bone = torch.ones((n, 1), dtype = torch.float32).to(X.device)
    H = (torch.eye(n) - torch.ones((n,n), dtype = torch.float32) / n).to(X.device)
    

    K = rbf_dot_torch(X, X, width_x)
    L = rbf_dot_torch(Y, Y, width_y)

    Kc = torch.mm(torch.mm(H, K), H)
    Lc = torch.mm(torch.mm(H, L), H)

    testStat = torch.sum(Kc.t() * Lc) / n

    return testStat




if __name__=='__main__':
	c_dim = 5
	s_dim = 5
	c = np.random.randn(1000, c_dim)
	s = np.random.randn(1000, s_dim)
    
	print(hsic_gam_torch(torch.tensor(c), torch.tensor(s)))