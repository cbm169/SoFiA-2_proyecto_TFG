/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (statistics_dbl.h) - Source Finding Application          ///
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


#ifndef STATISTICS_dbl_H
#define STATISTICS_dbl_H

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
int contains_nan_dbl(const double *data, const size_t size);

// Maximum and minimum
void max_min_dbl(const double *data, const size_t size, double *value_max, double *value_min);
double max_dbl(const double *data, const size_t size);
double min_dbl(const double *data, const size_t size);

// Sum and mean
double summation_dbl(const double *data, const size_t size, const bool mean);
double sum_dbl(const double *data, const size_t size);
double mean_dbl(const double *data, const size_t size);

// N-th moment
double moment_dbl(const double *data, const size_t size, unsigned int order, const double value);
void moments_dbl(const double *data, const size_t size, const double value, double *m2, double *m3, double *m4);

// Standard deviation
double std_dev_dbl(const double *data, const size_t size);
double std_dev_val_dbl(const double *data, const size_t size, const double value, const size_t cadence, const int range);

// N-th-smallest element
double nth_element_dbl(double *data, const size_t size, const size_t n);

// Median and MAD
double median_dbl(double *data, const size_t size, const bool fast);
double median_safe_dbl(double *data, const size_t size, const bool fast);
double mad_dbl(double *data, const size_t size);
double mad_val_dbl(const double *data, const size_t size, const double value, const size_t cadence, const int range);

// Robust and fast noise measurement
double robust_noise_dbl(const double *data, const size_t size);
double robust_noise_in_region_dbl(const double *data, const size_t nx, const size_t ny, const size_t x1, const size_t x2, const size_t y1, const size_t y2, const size_t z1, const size_t z2);

// Gaussian fit to histogram
size_t *create_histogram_dbl(const double *data, const size_t size, const size_t n_bins, const double data_min, const double data_max, const size_t cadence);
double gaufit_dbl(const double *data, const size_t size, const size_t cadence, const int range);

// Skewness and kurtosis
void skew_kurt_dbl(const double *data, const size_t size, double *skew, double *kurt);
double skewness_dbl(const double *data, const size_t size);
double kurtosis_dbl(const double *data, const size_t size);

// 1D boxcar filter
void filter_boxcar_1d_dbl(double *data, double *data_copy, const size_t size, const size_t filter_radius);

// 2D Gaussian filter
void filter_gauss_2d_dbl(double *data, double *data_copy, double *data_row, double *data_col, const size_t size_x, const size_t size_y, const size_t n_iter, const size_t filter_radius);

// Polynomial fitting
void shift_and_subtract_dbl(double *data, const size_t size, const size_t shift);


// ----------------- //
// Utility functions //
// ----------------- //

// Filter size and iterations for Gaussian function
void optimal_filter_size_dbl(const double sigma, size_t *filter_radius, size_t *n_iter);

#endif
