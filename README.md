Policy Optimization Dynamic Bayesian Network
===========
[![zheng](https://img.shields.io/badge/Author-Zheng.H-yellow)](https://zhenghuazx.github.io/hua.zheng/)

PABN is an library for biopharmaceutical manufacturing simulation, modeling and policy optimization in Python. It utilizes the `numpy` and `scipy`
for its core routines to give high performance solving of model training, policy optimization and differential equations, currently
including:

1. Bayesian network modeling and training (R-code)
2. Policy optimization in dynamic Bayesian network (liboptpy)
    * Shapley value based factor importance
    * bioprocess simulator
    * policy gradient algorithm
3. deep determininistic policy gradient (notebook/ddpg-fermentation.ipynb)



The algorithms and visualizations used in this package came primarily out of research in Wei Xie' lab at the Washington Northeastern. If you use PABN in your research we would appreciate a citation to the appropriate paper [INFORMS Journal of Computing (accepted)]([http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions](https://arxiv.org/pdf/2105.06543.pdf). 
