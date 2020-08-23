# Solar Magnetic Field Estimated from the Photospheric Continuum Image with Machine Learning Method 

This repo is the official implementation for our paper: Solar Magnetic Field Estimated from the Photospheric Continuum Image with Machine Learning Method. [[Paper]](https://github.com/FengTaoAI/test/) <br>

![image](https://github.com/Fonnn/test/blob/master/images/visualization.jpeg)


The repo is still under construction, current version provides,

* Trained models <br>
* The full inference and visualization <br>
* Samples of test image <br>
* Samples of test results <br>

### Dependencies

This code is tested on a Ubuntu 16.04 machine with a GTX 2080Ti GPU, with the following dependencies,

* Python 3.7 <br>
* PyTorch=1.3.0 <br>
* astropy=4.0 <br>
* matplotlib, numpy <br>

This repo may be able to be used in the GPU environment, but enough GPU memory is needed

### Folder Structure

* The folder ```./input```: continuum images for test
* The folder ```./output```: estimated abs(Br) and Bp images
* The folder ```./target```: Br, Bt, Bp from inversion (optional)
* The folder ```./model```:   two trained models for estimating abs(Br) and Bp
* The folder ```./resultsamples```:  some jpgs and mp4 samples for visualization

### Usage

Use trained model to test, run

```
python demo.py
```

The data and trained model can be found in the folder ```./input```, ```./model```


