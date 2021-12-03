# Pandora2D

<h1 align="center">
<img src="./doc/sources/Images/pandora2d_logo.png"/>
</h1>

<h4 align="center">Pandora2d  is a tool based on [Pandora](https://github.com/CNES/Pandora) to provide disparity maps
for images pairs with a combination of vertical and horizontal stereo.</h4>

<p align="center">
  <a href="#install">Example of use</a> •
  <a href="#install">Install</a> •
  <a href="#firststep">First Step</a> •
  <a href="#customize">Customize</a> •
  <a href="#credits">Credits</a> •
  <a href="#related">Related</a> •
  <a href="#references">References</a>
</p>

## Example of use :

* Not-aligned Sentinel2 images from Ouarzazate's Solar Central.

Before Pandora2D   |  After Pandora2D
:-----------------:|:----------------:
![](./doc/sources/Images/avant_recalage.gif)  |  ![](./doc/sources/Images/apres_recalage.gif)


## Install

Pandora2D is available on Pypi and can be installed by:

```bash
    #install pandora2d latest release
    pip install pandora2d
````

## First step

Pandora2d requires a `config.json` to declare the pipeline and the pair of images to process.
Download our data sample to start right away !

- [maricopa's pair with combination of vertical and horizontal stereo](https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/images/maricopa.zip)
- [a configuration file](https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/json_conf_files/a_basic_pipeline.json)

```bash
    # Images pairs with a combination of vertical and horizontal stereo
    wget https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/images/maricopa.zip
    # Config file
    wget https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/json_conf_files/a_basic_pipeline.json
    #uncompress data
    unzip maricopa.zip
    # run Pandora2d
    pandora2d a_basic_pipeline.json output_dir

    # The columns disparity map is saved in  "./res/columns_disparity.tif"
    # The row disparity map is saved in  "./res/row_disparity.tif"
```

## To go further

To create your own coregistration pipeline and choose among the variety of
algorithms we provide, please consult :ref `userguide`.

You will learn:

    * which steps you can use and combine
    * how to quickly set up a Pandora2D pipeline

## Credits

Our data test sample is based on the Peps Sentinel2 website (by CNES).

## Related

* [Pandora](<https://github.com/cnes/pandora>) - stereo matching framework