# SoFiA 2

This is version 2 of the HI Source Finding Application (SoFiA). SoFiA 2 is a reimplementation of the original SoFiA code in plain C. It is intended for use in HI data analysis pipelines and will be developed and maintained in parallel to SoFiA 1.x. While SoFiA 2 is currently under active development, a **stable version** is already available at this point in time and can be used in production mode. In addition, users will still be able to use **SoFiA 1.x** (https://github.com/SoFiA-Admin/SoFiA) for processing their data.

## Improvements in SoFiA 2

* Due to the use of C instead of Python, SoFiA 2 is significantly faster than SoFiA 1.x (by up to a factor of 2.5).
* SoFiA 2 requires significantly less memory than SoFiA 1.x (down from > 5 × cube size to ~ 2.25 × cube size).
* SoFiA 2 currently has only a single external dependency (WCSLIB) and should therefore compile and run on any machine with a Linux or Unix operating system and the GCC compiler and WCSLIB installed.

## Installation

Please ensure that the **GNU C compiler** (GCC) and **WCSLIB** (https://www.atnf.csiro.au/people/mcalabre/WCS/) are installed on your machine, before downloading and extracting the SoFiA 2 **source code** into a directory of your choice. You may want to first check if WCSLIB is available from your operating system’s software repository (`wcslib-dev` package) before you attempt to install it by hand.

Once WCSLIB is installed, execute the `compile.sh` script to compile SoFiA 2 using the GCC compiler. Please ensure that you follow the **instructions** printed at the end of the compilation process to finalise the installation. If a compiler error related to WCSLIB shows up, please ensure that WCSLIB is installed in a standard location where it can be found by the GCC.

## Documentation

While documentation for SoFiA 2 is still limited at this stage, an overview of all control parameters as well as a copy of the SoFiA 2 User Manual can be found on the wiki at https://github.com/SoFiA-Admin/SoFiA-2/wiki.

## Feedback

Should you decide to run SoFiA 2 on your own data, we would welcome any feedback on how well SoFiA 2 works for you and what improvements could be made. If you have a GitHub account, you can directly create a new issue (https://github.com/SoFiA-Admin/SoFiA-2/issues/new) on GitHub for questions, feature requests or bug reports. Alternatively, please feel free to directly contact the lead programmer, Tobias Westmeier, via e-mail at `tobias.westmeier (at) uwa.edu.au` to provide feedback on your experience with SoFiA 2. Note that the main purpose of SoFiA 2 is to facilitate the processing of HI data from SKA precursor surveys, and we are unlikely to be able to accommodate requests for additional features beyond this primary scope.

## Copyright and licence

Copyright (C) 2020 Tobias Westmeier

SoFiA 2 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License  along with this program. If not, see http://www.gnu.org/licenses/.
