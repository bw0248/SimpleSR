# SimpleSR
![](https://img.shields.io/badge/Version-0.1-critical)
![](https://img.shields.io/badge/Python-3.8-informational)
![](https://img.shields.io/badge/Tensorflow-2.0%2B-informational)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A simple library for Deep Learning based Single Image Super-Resolution  
*Note: The project is still in an early stage and under active development, things might not work or possibly break with future releases.
Feel free to open an issue if you encounter a problem.*

## Pictures of results
* todo

## Features
* Implementations of popular Super-Resolution network architectures:
    - SRResnet
    - SRGAN (standard GAN)
    - RRDB
    - ESRGAN (relativistic average GAN)
* Perceptual loss based on VGG19 network
    - easily combine multiple VGG Layers for feature loss computation
    - compute feature loss before or after VGG activation functions
* Data pipeline based on tf.data api for more efficient training
    - optionally crop a number of patches from training images for further increased efficiency
    - support for various augmentations during preprocessing
    - option to degrade low-res training samples with jpg noise to learn noise removal
* Easy configuration of Model/Training via single YAML file
* Evaluate/compare models and produce image grids of your personal test pictures

## Installation
* todo

## Datasets
For experimentation typical Super-Resolution datasets can be found here:

<table>
    <tr>
        <th>Type</th>
        <th>Dataset</th>
        <th>Link</th>
    </tr>
    <tr>
        <th rowspan="3">Training</th>
        <td>Flickr2K</td>
        <td><a href="https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar">direct download from EDSR team</a></td>
    </tr>
    <tr>
        <td>Div2K</td>
        <td><a href="https://data.vision.ee.ethz.ch/cvl/DIV2K/">download from ETHZ</a></td>
    </tr>
    <tr>
        <td>CelebA-HQ</td>
        <td><a href="https://drive.google.com/drive/folders/11Vz0fqHS2rXDb5pprgTjpD7S2BAJhi1P">download from google drive</a></td>
    </tr>
    <tr>
        <th rowspan="6">Testing</th>
        <td>Set5</td>
        <td rowspan="6"><a href="http://vllab.ucmerced.edu/wlai24/LapSRN/results/SR_testing_datasets.zip">direct download from LapSRN team</a></td>
    </tr>
    <tr>
        <td>Set14</td>
    </tr>
    <tr>
        <td>BSDS100</td>
    </tr>
    <tr>
        <td>Urban100</td>
    </tr>
    <tr>
        <td>Manga109</td>
    </tr>
    <tr>
        <td>Historical</td>
    </tr>
</table>
 
## Getting started
* todo

## Todo

* [ ] dataset downloader
* [ ] better support for user defined architectures/loss functions
* [ ] network interpolation for combining different networks
* [ ] support for TTFRecords 
* [ ] build documentation with sphinx
* [ ] package for pypi

## Known Issues

* Sometimes the RRDB network after training successfully 
for a few hundred thousand iterations degrades very quickly and only produces
completely black images afterwards

## Links

### Other Super-Resolution Projects that provided inspiration for SimpleSR

* [BasicSR](https://github.com/xinntao/BasicSR) (SR toolbox from ESRGAN paper authors)
* [Official ESRGAN implemenation](https://github.com/xinntao/ESRGAN) (from ESRGAN paper authors)
* [Idealo Image Super-Resolution](https://github.com/idealo/image-super-resolution) 
(project to upscale images from idealo.de product catalog)
* [Martin Krasser - Super Resolution](https://github.com/krasserm/super-resolution)
(Tensorflow 2.0 implementation of EDSR, WDSR and SRGAN)

### Papers

* [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (Ledig et al.)](https://arxiv.org/abs/1609.04802)
* [ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (Xintao Wang et al.)](https://arxiv.org/abs/1809.00219)
* [Deep Learning for Image Super-resolution:A Survey (Zhihao Wang et al.)](https://arxiv.org/pdf/1902.06068.pdf)
* [A Deep Journey into Super-resolution: A Survey (Anwar et al.)](https://arxiv.org/pdf/1904.07523.pdf)
