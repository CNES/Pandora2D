# Changelog

## 0.3.0 (June 2024)

### Added
- Adding configuration file with dichotomy step. [#132]
- Add warning documentation about stored disparity maps when step is different from 1. [#140]
- Add warning documentation about pandora2d maturity. [#129]

### Fixed
- Fix estimation check_conf method. [#115]
- Fix plot disparity maps with correct dataset on the notebook. [#139]
- Fix update image path on README.md file. [#135]

### Changed
- Add ROI in test_dichotomy_execution. [#134]
- Update version copyright. [#133]
- Update README.md file. [#136]

## 0.3.0a1 (June 2024)

### Added

- Adding the dichotomy module in refinement step. [#64]
- Adding margins in the dichotomy module. [#70]
- Adding optical flow and phase cross correlation. [#52]
- Adding the validation plan. [#45]
- Adding ROI coordinates in the disparity maps datasets. [#98]
- Adding note about odd window size in documentation. [#100]
- Adding matching cost score in disparity maps. [#77]
- Adding shift image method for subpix. [#109]
- Adding MCCNN similarity measure. [#2]
- Adding dichotomy refinement method. [#65]
- Adding tests about ROI in check_configuration (validation plan). [#91]
- Adding subpix parameter in matching_cost step. [#84]
- Adding disparity margins in cost volumes. [#103]
- Adding interpolation filter module. [#101]
- Adding configuration tests (validation plan). [#89]
- Adding bicubic module in interpolation filter. [#104] 
- Adding georeferencing to disparity maps. [#110]
- Adding missing tests detected with coverage. [#121]
- Addind documentation for dichotomy. [#68]

### Fixed

- Fix the github CI by adding the pytest_monitor dependency.
- Fix interpolation error message when the disparity interval is smaller than 5. [#96]
- Remove ROI margins in resulting disparity maps. [#97]
- Update notebooks after evolution of run and run_prepare methods. [#93]
- Fix notebook tests and clean-doc in Makefile.
- Fix ROI margins in configuration. [#106]
- Inverse row and column invalid disparity in optical flow. 
- Fix FAQ and before coding documentation. [#112]
- Transmission of ROI in Pandora cost volume allocation. [#107]
- Fix cost volume coordinates when using a ROI and a step different from 1. [#108]
- Add ValueError message when disparity range is out of the image. [#111]
- Remove get_global_margins method. [#117]
- Fix window size equal to one with optical flow. [#118]
- Fix correlation score and georeferencing with ROI. [#122] 


### Changed

- Update matching cost check_conf. [#125]
- Update dichotomy refinement method. [#105]

## 0.2.0 (February 2024)

### Added

- Adding roi in user configuration file. [#23]
- Adding step in uset configuration file. [#22]
- Added margin calculation for the treatment chain with ROI image. [#27]
- Added a notebook to demonstrate the use of the roi and the step. [#24]
- Added create_datasets_from_inputs method to read image. [#34]
- Added check_datasets to finish checking everything after reading the images. [#37]
- Added row step and col step in matching_cost. [#26]

### Fixed

- Force python version >= 3.7. [#7]
- Fix the code development tools. [#39]
- Adaptation to new pandora compute_cost_volume method API. [#42]
- Fix with new pandora check_dataset method. [#43]
- Fix with new pandora check_disparities method. [#44]
- Lighter interfaces containing disparities. [#35]
- Correction doc. [#47]
- Adapt with disparity_source attribute in xarray dataset. [#49]
- Rename doc sources to docs source. [#51]
- Replacing sys.exit by raises (pandora changes). [#58]
- Fix readthedocs. [#50]
- Fix warning for xarray.Dataset.dims. [#60]
- Update with new validity_mask from pandora. [#62]

### Changed

- Adapt to setuptools 67. [#10]
- Adapt to pandora version 1.5.0 . [#11]
- Rename check_json to check_configuration. [#29]
- Update pandora image read method call, read_img to create_dataset_from_inputs. [#31]
- New format for disparity in the user configuration file. [#28]
- Update user configuration file with new keys : "left" & "right". [#33]
- Move allocate_cost_volume. [#56]
- Replacing sys.exit by raises. [#59]
- Update of the minimal version for python. [#61]
- Alignment of margins with those of pandora. [#67]

## 0.1.0 (February 2022)

### Added

- First version of Pandora2D tools


