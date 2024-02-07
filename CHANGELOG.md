# Changelog

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


