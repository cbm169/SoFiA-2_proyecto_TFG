/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (Catalog.h) - Source Finding Application                 ///
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

#ifndef CATALOG_H
#define CATALOG_H

#include <stdint.h>
#include "common.h"
#include "Source.h"

#define CATALOG_COLUMN_WIDTH 14

typedef enum {CATALOG_FORMAT_ASCII, CATALOG_FORMAT_XML, CATALOG_FORMAT_SQL} file_format;


// ----------------------------------------------------------------- //
// Class 'Catalog'                                                   //
// ----------------------------------------------------------------- //
// The purpose of this class is to provide a structure for storing   //
// and handling source catalogues as a simple list of objects of     //
// class 'Source'.                                                   //
// ----------------------------------------------------------------- //

typedef CLASS Catalog Catalog;

// Constructor and destructor
PUBLIC Catalog *Catalog_new           (void);
PUBLIC void     Catalog_delete        (Catalog *self);

// Public member functions
PUBLIC void     Catalog_add_source    (Catalog *self, Source *src);
PUBLIC Source  *Catalog_get_source    (const Catalog *self, const size_t index);
PUBLIC size_t   Catalog_get_index     (const Catalog *self, const Source *src);
PUBLIC bool     Catalog_source_exists (const Catalog *self, const Source *src, size_t *index);

PUBLIC size_t   Catalog_get_size      (const Catalog *self);

PUBLIC void     Catalog_save          (const Catalog *self, const char *filename, const file_format format, const bool overwrite);

#endif
