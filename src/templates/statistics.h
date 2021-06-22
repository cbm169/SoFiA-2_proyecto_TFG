/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (statistics_SFX.h) - Source Finding Application          ///
/// Copyright (C) 2020 Tobias Westmeier                                  ///
/// ____________________________________________________________________ ///
///                                                                      ///
/// Address:  Tobias Westmeier                                           ///
///           ICRAR M468                                                 ///
///           The University of Western Australia                        ///
///           35 Stirling Highway                                        ///
///           Crawley WA 6009                                            ///
///           Australia                                                  ///
///                                                                      ///
/// E-mail:   tobias.westmeier [at] uwa.edu.au                           ///
/// ____________________________________________________________________ ///
///                                                                      ///
/// This program is free software: you can redistribute it and/or modify ///
/// it under the terms of the GNU General Public License as published by ///
/// the Free Software Foundation, either version 3 of the License, or    ///
/// (at your option) any later version.                                  ///
///                                                                      ///
/// This program is distributed in the hope that it will be useful,      ///
/// but WITHOUT ANY WARRANTY; without even the implied warranty of       ///
/// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the         ///
/// GNU General Public License for more details.                         ///
///                                                                      ///
/// You should have received a copy of the GNU General Public License    ///
/// along with this program. If not, see http://www.gnu.org/licenses/.   ///
/// ____________________________________________________________________ ///
///                                                                      ///


// WARNING: This is a template that needs to be instantiated before use.
//          Do not edit template instances, as they are auto-generated!


#ifndef STATISTICS_SFX_H
#define STATISTICS_SFX_H

#include <stdbool.h>
#include "common.h"

// -------------------------- //
// Settings for boxcar filter //
// -------------------------- //
#define BOXCAR_MIN_ITER 3
#define BOXCAR_MAX_ITER 6



// -------------------- //
// Statistics functions //
// -------------------- //

// Check if array contains NaN
int contains_nan_SFX(const DATA_T *data, const size_t size);

// Maximum and minimum
void max_min_SFX(const DATA_T *data, const size_t size, DATA_T *value_max, DATA_T *value_min);
DATA_T max_SFX(const DATA_T *data, const size_t size);
DATA_T min_SFX(const DATA_T *data, const size_t size);

// Sum and mean
double summation_SFX(const DATA_T *data, const size_t size, const bool mean);
double sum_SFX(const DATA_T *data, const size_t size);
double mean_SFX(const DATA_T *data, const size_t size);

// N-th moment
double moment_SFX(const DATA_T *data, const size_t size, unsigned int order, const double value);
void moments_SFX(const DATA_T *data, const size_t size, const double value, double *m2, double *m3, double *m4);

// Standard deviation
double std_dev_SFX(const DATA_T *data, const size_t size);
double std_dev_val_SFX(const DATA_T *data, const size_t size, const double value, const size_t cadence, const int range);

// N-th-smallest element
DATA_T nth_element_SFX(DATA_T *data, const size_t size, const size_t n);

// Median and MAD
DATA_T median_SFX(DATA_T *data, const size_t size, const bool fast);
DATA_T median_safe_SFX(DATA_T *data, const size_t size, const bool fast);
DATA_T mad_SFX(DATA_T *data, const size_t size);
DATA_T mad_val_SFX(const DATA_T *data, const size_t size, const DATA_T value, const size_t cadence, const int range);

// Robust and fast noise measurement
DATA_T robust_noise_SFX(const DATA_T *data, const size_t size);
DATA_T robust_noise_in_region_SFX(const DATA_T *data, const size_t nx, const size_t ny, const size_t x1, const size_t x2, const size_t y1, const size_t y2, const size_t z1, const size_t z2);

// Gaussian fit to histogram
size_t *create_histogram_SFX(const DATA_T *data, const size_t size, const size_t n_bins, const DATA_T data_min, const DATA_T data_max, const size_t cadence);
DATA_T gaufit_SFX(const DATA_T *data, const size_t size, const size_t cadence, const int range);

// Skewness and kurtosis
void skew_kurt_SFX(const DATA_T *data, const size_t size, double *skew, double *kurt);
double skewness_SFX(const DATA_T *data, const size_t size);
double kurtosis_SFX(const DATA_T *data, const size_t size);

// 1D boxcar filter
void filter_boxcar_1d_SFX(DATA_T *data, DATA_T *data_copy, const size_t size, const size_t filter_radius);

// 2D Gaussian filter
void filter_gauss_2d_SFX(DATA_T *data, DATA_T *data_copy, DATA_T *data_row, DATA_T *data_col, const size_t size_x, const size_t size_y, const size_t n_iter, const size_t filter_radius);

// Polynomial fitting
void shift_and_subtract_SFX(DATA_T *data, const size_t size, const size_t shift);


// ----------------- //
// Utility functions //
// ----------------- //

// Filter size and iterations for Gaussian function
void optimal_filter_size_SFX(const double sigma, size_t *filter_radius, size_t *n_iter);

#endif
