/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (Array_SFX.h) - Source Finding Application               ///
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


#ifndef ARRAY_SFX_H
#define ARRAY_SFX_H

#include "common.h"


// ----------------------------------------------------------------- //
// Class 'Array_SFX'                                                 //
// ----------------------------------------------------------------- //
// The purpose of this class is to provide a convenient way to store //
// multiple values of a specific type in an array-like structure. A  //
// new array can either be of a given size and empty (using the      //
// standard constructor) or provided with a list of comma-separated  //
// values that will be stored in the array and used to determine its //
// size using the alternative constructor).                          //
// ----------------------------------------------------------------- //

typedef CLASS Array_SFX Array_SFX;

// Constructor and destructor
PUBLIC Array_SFX    *Array_SFX_new      (const size_t size);
PUBLIC Array_SFX    *Array_SFX_new_str  (const char *string);
PUBLIC Array_SFX    *Array_SFX_copy     (const Array_SFX *source);
PUBLIC void          Array_SFX_delete   (Array_SFX *self);

// Methods
PUBLIC const DATA_T *Array_SFX_get_ptr  (const Array_SFX *self);
PUBLIC size_t        Array_SFX_get_size (const Array_SFX *self);
PUBLIC Array_SFX    *Array_SFX_push     (Array_SFX *self, const DATA_T value);
PUBLIC DATA_T        Array_SFX_get      (const Array_SFX *self, const size_t index);
PUBLIC Array_SFX    *Array_SFX_set      (Array_SFX *self, const size_t index, const DATA_T value);
PUBLIC Array_SFX    *Array_SFX_add      (Array_SFX *self, const size_t index, const DATA_T value);
PUBLIC Array_SFX    *Array_SFX_cat      (Array_SFX *self, const Array_SFX *source);

#endif
