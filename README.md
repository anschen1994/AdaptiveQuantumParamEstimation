# AdaptiveQuantumParamEstimation
=========================================================
This is code in our paper url... \\ 

We raise a general scheme to do quantum parameter estimation adaptively. People can design their custom dynamics as the example we show in model.py. This code is contributed by Yu Chen(anschen@link.cuhk.edu.hk) and Hongzhen Chen.

------------------------------------------------
For magnetometer, we take parameter:
<img src = "http://latex.codecogs.com/gif.latex? \theta=\frac{\pi}{4}, B=0.1, \gamma=0.05, t=5, samples=1000, m=10, N=10"/>

For gyroscope, we take parameter:
<img src = "http://latex.codecogs.com/gif.latex? \Omega = 0.3, B=0.1, t=5, \gamma=0.05, samples=1000, m=10, N=10"/>

The adaptive process is like:  
Repeat:  
    1) Make a guess estimator <img src = "http://latex.codecogs.com/gif.latex?\hat{\theta}(n)" />.  
    2) Design control rule <img src = "http://latex.codecogs.com/gif.latex?\mathcal{C}(\hat{\theta})" /> based on your theory.  
    3) Do $m$ meausurements, and you obtain some results <img src = "http://latex.codecogs.com/gif.latex?\{x_i\}" />.  
    4) Update your estimator by maximizing likelihood <img src = "http://latex.codecogs.com/gif.latex?\hat{\theta}(n+1)=\arg\max_{\theta}L(\theta;\{x_i\})" />.
