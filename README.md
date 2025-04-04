<h1 align="center">
<a href="https://pandora2d.readthedocs.io/?badge=latest"><img src="https://raw.githubusercontent.com/CNES/Pandora2D/master/docs/source/Images/logo/logo_pandora2d.png?inline=false" alt="Pandora2D coregistration Framework" width="432"></a>
</h1>

<h4 align="center">Pandora2d  is a tool based on <a href="https://github.com/CNES/Pandora">Pandora<a/> to provide disparity maps
for images pairs with a combination of vertical and horizontal stereo.</h4>

<p align="center">
  <a><img src="https://github.com/CNES/Pandora2D/actions/workflows/pandora2d_ci.yml/badge.svg?branch=master"></a>
  <a href='https://pandora2d.readthedocs.io/?badge=latest'><img src='https://readthedocs.org/projects/pandora2d/badge/?version=latest' alt='Documentation Status' /></a>
  <a href="https://opensource.org/licenses/Apache-2.0/"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://codecov.io/gh/CNES/Pandora2D"><img src="https://codecov.io/gh/CNES/Pandora2D/branch/master/graph/badge.svg"></a>
</p>

<p align="center">
  <a href="#example-of-use">Example of use</a> •
  <a href="#install">Install</a> •
  <a href="#first-step">First Step</a> •
  <a href="#to-go-further">To go further</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> 
</p>

## Example of use

* Not-aligned Sentinel2 images from Ouarzazate's Solar Central.

Before Pandora2D   |  After Pandora2D
:-----------------:|:----------------:
![](https://raw.githubusercontent.com/CNES/Pandora2D/master/docs/source/Images/avant_recalage.gif?inline=false)  |  ![](https://raw.githubusercontent.com/CNES/Pandora2D/master/docs/source/Images/apres_recalage.gif?inline=false)


## Install

Pandora2D is available on Pypi and can be installed by:

```bash
    # Upgrade your pip by running:
    pip install --upgrade pip
    # Install pandora2d latest release
    pip install pandora2d
````

## Quick start

Pandora2d requires a `config.json` to declare the pipeline and the pair of images to process.
Download our data sample to start right away !

- [maricopa's pair with combination of vertical and horizontal stereo](https://raw.githubusercontent.com/CNES/Pandora2D/master/data_samples/images/maricopa.zip)
- [a configuration file](https://raw.githubusercontent.com/CNES/Pandora2D/master/data_samples/json_conf_files/a_basic_pipeline.json)

```bash
    # Images pairs with a combination of vertical and horizontal stereo
    wget https://raw.githubusercontent.com/CNES/Pandora2D/master/data_samples/images/maricopa.zip
    # Config file
    wget https://raw.githubusercontent.com/CNES/Pandora2D/master/data_samples/json_conf_files/a_basic_pipeline.json
    # Uncompress data
    unzip maricopa.zip
    # run Pandora2d
    pandora2d a_basic_pipeline.json output_dir

    # The columns disparity map is saved in  "./output_dir/columns_disparity.tif"
    # The row disparity map is saved in  "./output_dir/row_disparity.tif"
```

## To go further

To create your own coregistration pipeline and choose among the variety of
algorithms we provide, please consult [our online documentation](https://pandora2d.readthedocs.io).

You will learn:

* which steps you can use and combine
* how to quickly set up a Pandora2D pipeline

## Credits

Our data test sample contains modified 'Copernicus Sentinel data [2021]', provided by the Peps Sentinel2 website (CNES).


## Related

* [Pandora](<https://github.com/cnes/pandora>) - stereo matching framework
