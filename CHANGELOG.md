# Changelog

## 0.5.0 (May 2025)

### Added

- Added P2D_RIGHT_DISPARITY_OUTSIDE in validity dataset. [#268]
- Added P2D_INVALID_MASK_LEFT & P2D_INVALID_MASK_RIGHT in validity dataset. [#269]
- Added P2D_LEFT_NODATA & P2D_RIGHT_NODATA in validity dataset. [#267]
- Added P2D_PEAK_ON_EDGE in validity dataset. [#270]
- Added validity criteria documentation. [#283]


### Fixed

- Fixed functional and resource tests execution for dichotomy. [#248]
- Fixed errors in log. [#285]
- Fixed errors in documentation. [#278]
- Fixed mutual information with nb_bins greater than NB_BINS_MAX. [#286]
- Fixed estimation bug with input disparities. [#276]

### Changed

- Moved cost volumes allocation in matching cost prepare step. [#280]
- Used the criteria dataarray in WTA. [#272]
- Updated pandora version in pyproject.toml. [#278]
- Removed P2D_DISPARITY_UNPROCESSED in output validity.tif. [#301]


## 0.5.0a1 (March 2025)

### Added
- Created criteria data array in the state machine. [#194]
- Added dichotomy CPP module. [#211]
- Added base matching cost class. [#208]
- Added abstract filter CPP class. [#212]
- Added bicubic module CPP in interpolation filter. [#215]
- Added binding for bicubic CPP. [#216]
- Added histogram1D and histogram2D classes. [#223]
- Added entropy class. [#224]
- Added Python class for mutual information. [#221]
- Added interpolation method for the right image. [#226]
- Added Python class for cardinal sine CPP. [#213]
- Added margin notebook. [#235]
- Added binding for cardinal sine CPP. [#216]
- Added dependencies file for Meson. [#238]
- Added compute cost volume method CPP. [#227]
- Added binding for compute cost volume. [#229]
- Added refinement method using dichotomy CPP. [#217]
- Added performance tests for mutual information. [#231]
- Added attributes JSON file. [#242]
- Added documentation for mutual information. [#230]
- Added peak on edge criteria. [#255]
- Added criteria data array on cost volume. [#243]
- Added validity mask output. [#172]
- Added error message in documentation for mask with step. [#261]
- Added wheel for cpp code. [#237]

### Fixed
- Fixed dichotomy with a step different from 1. [#199]
- Improved margin management using ROI. [#196]
- Refactored imports for Sphinx to fix documentation build. [#220]
- Fixed subpixel on dichotomy CPP. [#218]
- Fixed installation on Read the Docs. [#236]
- Fixed typo for mutual information. [#258]
- Fixed memory leak for dichotomy. [#252]
- Fixed histogram2D computation when the number of bins is too large. [#257]
- Fixed resource tests. [#249]

### Changed
- Separated margin types: image reading margins and disparity margins. [#197]
- Clarified message when input disparities are not a dictionary. [#198]
- Updated tests. [#201]
- Refactored criteria data array allocation. [#202]
- Updated matching cost tests. [#203]
- Updated refinement tests. [#204]
- Updated disparity tests. [#205]
- Inverted disparity arguments on cost_volume. [#206]
- Renamed dichotomy method in dichotomy_python. [#210]
- Updated ratio display documentation. [#232]
- Updated Sonar settings.
- Updated type for disparity on Xarray to float32. [#222]
- Moved from setuptools to Meson for Python. [#233]
- Updated Makefile for CPP code. [#234]
- Updated save dataset method. [#240]
- Updated documentation and notebook with the new dichotomy version. [#218]
- Updated CLI with output parameters. [#246]
- Updated output directories. [#241]
- Updated input path in user configuration. [#244]
- Disabled CPP tests by default. [#260]
- Updated Flag type for criteria. [#256]
- Refactored CPP code. [#250]
- Fixed interface name. [#264]
- Standardized types in configuration tables in documentation. [#263]
- Removed interpolation method. [#274]


## 0.4.0 (October 2024)

### Added
- Add unit tests for optical flows. [#114]
- Subpix taken into account in the dichotomy. [#148]
- Adding requirement numbers to the test docstring. [#143]
- Add requirement on html-report test. [#167]
- Variable initial disparity added to the configuration file. [#76]
- Adding cardinal sine module in interpolation filter. [#146]
- Mask added to the configuration file. [#157]
- Variable disparity taken into account in matching cost step. [#152]
- Add constant.py and criteria.py files for masks. [#159]
- Add right-disparity-outside criterion. [#162]
- Variable disparity taken into account in dichotomy. [#154]
- Add accuracy tests for dichotomy. [#126]
- Variable disparity taken into account in optical flow. [#158]
- Add first resource tests. [#174]
- Setting up disparity grids at inputs. [#165]
- Add left_nodata and right_nodata criteria. [#160]
- Add left_invalid and right_invalid criteria. [#161]
- Add criteria dataarray. [#163]
- Add profiling. [#175]

### Fixed
- Fix the use of a step with the optical flow method in the refinement step. [#119]
- Fix ROI coordinates when the first point is within the margin. [#142]
- Fix sphinx errors. [#168]
- Remove np.inf on cost volume. [#170]

### Changed
- State machine callback changed from after to before. [#144]
- Update pylint version. [#153]
- Documentation updated with new parameters for variable initial disparity. [#150]
- Update numpy version. [#145]
- Removal of disparity grids in the state machine. [#171]
- Removal of the disparity condition with the interpolation step. [#169]
- Update dichotomy documentation. [#166]
- Pixel size output updated as a function of step size. [#164]

 
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


