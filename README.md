# SGD with large step sizes learns sparse features

**Maksym Andriushchenko, Aditya Varre, Loucas Pillaud-Vivien, Nicolas Flammarion (EPFL)**

**Paper:** [TODO]()

<!-- <p align="center"><img src="images/twitter.gif" width="400" /></p> -->
<p align="center"><video src="images/twitter.mp4" controls="controls" style="max-width: 500px;"></video></p>


## Abstract
We showcase important features of the dynamics of the Stochastic Gradient Descent (SGD) in the training of neural networks. We present empirical observations that commonly used large step sizes (i) lead the iterates to jump from one side of a valley to the other causing *loss stabilization*, and (ii) this stabilization induces a hidden stochastic dynamics orthogonal to the bouncing directions that *biases it implicitly* toward simple predictors. Furthermore, we show empirically that the longer large step sizes keep SGD high in the loss landscape valleys, the better the implicit regularization can operate and find sparse representations. Notably, no explicit regularization is used so that the regularization effect comes solely from the SGD training dynamics influenced by the step size schedule. Therefore, these observations unveil how, through the step size schedules, both gradient and noise drive together the SGD dynamics through the loss landscape of neural networks. We justify these findings theoretically through the study of simple neural network models as well as qualitative arguments inspired from stochastic processes. Finally, this analysis allows to shed a new light on some common practice and observed phenomena when training neural networks.

<p align="center"><img src="images/fig1.png" width="900" /></p>


## Code
The exact code to reproduce all the reported experiments on simple networks is available in jupyter notebooks:
- `diag_nets.ipynb`: diagonal linear networks (also see `diag_nets_2d_loss_surface.ipynb` for loss surface visualizations).
- `fc_nets_1d_regression.ipynb`: two-layer ReLU networks on 1D regression problem.
- `fc_nets_two_layer.ipynb`: two-layer ReLU networks in a teacher-student setup (+ neuron movement visualization).
- `fc_nets_multi_layer.ipynb`: three-layer ReLU networks in a teacher-student setup.

For deep networks: see folder `deep_nets`. All the dependencies for deep networks are collected in the `Dockerfile`.

