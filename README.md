nsp (Non-linear signal processing)
=========
Sandbox for non-linear signal processing in Python (2.7) written for a better understanding of the complex methods. All materiel and syntax is inspired by the work of Lars Kai Hansen DTU for the 02657 non-linear signal processing course.

### Requirements
 - [NumPy](https://github.com/numpy/numpy), Scientific computing with Python.
 - [Matplotlib](https://github.com/matplotlib/matplotlib), A 2D plotting library.

### Examples
This is a timeseries prediction example using a Gaussian Process which is found in `example_gp.py`. Both the best log-likelihood and least-square fit is shown. A 95% confidence interval along the predictions is shown in the right figure.
<p align="center">
<img src="https://github.com/aaskov/nsp/blob/master/images/gp_example.png?raw=true" width="80%"/>
</p>

This is a two-class classification example using a Support Vector Machine which is found in `example_svm.py`. The algorihtm learns a set of support vectors that can be used for new (unseen) observations. The test result is shown in the right figure.
<p align="center">
<img src="https://github.com/aaskov/nsp/blob/master/images/svm_example.png?raw=true" width="80%"/>
</p>
