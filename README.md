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
3. Deep determininistic policy gradient (notebook/ddpg-fermentation.ipynb)
4. Yeast cell fermentation experimental data

## Bayesian network
Before optimizing the policy, it is important to train the Bayesian network model:
   * fermentation_BN_new2.R: use function `bioreactor` to generate the simulation data from Bioreactor simulator and run Gibbs sampler to learn posterior distribution of BN.
   * end-to-end_BN.R: the main difference from fermentation_BN_new2.R is that it adds a new simulation code ("purification after C1") to generate downstream process data and the resuslting Bayesian network is extended.
   
## Policy optimization
Then the policy optimization
- function projection(y): defines policy constraint
- training parameters:
     * patient: Number of epochs with no improvement after which training will be stopped.
     * window: moving window we consider the improvement
- max_iter: maximum number of iterations

Training script:
* liboptpy/run_fermentation.py: train the optimal policy based on the Bayesian network obtained in the previous step.
* liboptpy/run_end2end.py: train the optimal policy based on the Bayesian network in an end-to-end bioprocess (fermentation + purification)
* notebook/BN_MDP_sigma10_R100_unfixedBN_PGA.ipynb: jupyter notebook example
 
 ## Citations

The algorithms and visualizations used in this package came primarily out of research in Wei Xie' lab at the Washington Northeastern. If you use PABN in your research we would appreciate a citation to the appropriate paper [INFORMS Journal of Computing (accepted)]([http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions](https://arxiv.org/pdf/2105.06543.pdf). 
