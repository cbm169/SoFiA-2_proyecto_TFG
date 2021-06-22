/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (LinkerPar.h) - Source Finding Application               ///
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

#ifndef LINKERPAR_H
#define LINKERPAR_H

#include "common.h"
#include "Map.h"
#include "Matrix.h"
#include "Catalog.h"


// ----------------------------------------------------------------- //
// Class 'LinkerPar'                                                 //
// ----------------------------------------------------------------- //
// The purpose of this class is to provide a structure for storing   //
// and updating source parameters handled by the linker implemented  //
// in the class 'DataCube'.                                          //
// ----------------------------------------------------------------- //

typedef CLASS LinkerPar LinkerPar;

// Constructor and destructor
PUBLIC LinkerPar *LinkerPar_new          (const bool verbosity);
PUBLIC void       LinkerPar_delete       (LinkerPar *self);

// Public methods
PUBLIC size_t     LinkerPar_get_size     (const LinkerPar *self);
PUBLIC void       LinkerPar_push         (LinkerPar *self, const size_t label, const size_t x, const size_t y, const size_t z, const double flux, const unsigned char flag);
PUBLIC void       LinkerPar_pop          (LinkerPar *self);
PUBLIC void       LinkerPar_update       (LinkerPar *self, const size_t x, const size_t y, const size_t z, const double flux, const unsigned char flag);
PUBLIC void       LinkerPar_update_flag  (LinkerPar *self, const unsigned char flag);
PUBLIC size_t     LinkerPar_get_obj_size (const LinkerPar *self, const size_t label, const int axis);
PUBLIC size_t     LinkerPar_get_npix     (const LinkerPar *self, const size_t label);
PUBLIC void       LinkerPar_get_bbox     (const LinkerPar *self, const size_t label, size_t *x_min, size_t *x_max, size_t *y_min, size_t *y_max, size_t *z_min, size_t *z_max);
PUBLIC double     LinkerPar_get_flux     (const LinkerPar *self, const size_t label);
PUBLIC double     LinkerPar_get_rel      (const LinkerPar *self, const size_t label);
PUBLIC size_t     LinkerPar_get_label    (const LinkerPar *self, const size_t index);

PUBLIC Catalog   *LinkerPar_make_catalog (const LinkerPar *self, const Map *filter, const char *flux_unit);
PUBLIC void       LinkerPar_print_info   (const LinkerPar *self);

// Reliability filtering
PUBLIC Matrix    *LinkerPar_reliability  (LinkerPar *self, const double scale_kernel, const double fmin);
PUBLIC void       LinkerPar_rel_plots    (const LinkerPar *self, const double threshold, const double fmin, const Matrix *covar, const char *filename, const bool overwrite);

#endif
