# Solar Magnetic Field Estimated from the Photospheric Continuum Image with Machine Learning Method 

This repo is the official implementation for our paper: Solar Magnetic Field Estimated from the Photospheric Continuum Image with Machine Learning Method <br>

![image](https://github.com/jikaifan/IC2mag/blob/master/icons/visualization.jpeg)


The repo is still under construction, current version provides,

* Trained models <br>
* The full inference and visualization <br>
* Samples of test continuum image <br>
* Samples of test results <br>

### Dependencies

This code is tested on a Ubuntu 16.04 machine with a GTX 2080Ti GPU, with the following dependencies,

* Python 3.7 <br>
* PyTorch=1.3.0 <br>
* astropy=4.0 <br>
* matplotlib, numpy <br>

This repo is able to be used in the GPU environment, but enough GPU memory is needed

### Folder Structure

* The folder ```./input```: continuum images for model testing
* The folder ```./model```: two trained models for estimating abs(Br) and Bp
* The folder ```./output```: estimated abs(Br) and Bp images from our CNN models
* The folder ```./figure```: figure of output result
* The folder ```./target```: Br, Bt, Bp from inversion (optional)
* The folder ```./resultsamples```: some samples of test set results, include JPGs and MP4 movies

### Usage

Use trained model to test, run

```
python demo.py
```

The test images and trained models can be found in the folder ```./input```, ```./model```


