/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (String.c) - Source Finding Application                  ///
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

#ifndef STRING_H
#define STRING_H

#include "common.h"

typedef CLASS String String;


// ----------------------------------------------------------------- //
// Class 'String'                                                    //
// ----------------------------------------------------------------- //
// This class provides a convenient structure for storing and hand-  //
// ling strings. These strings are dynamic, meaning that they adjust //
// their memory allocation automatically and dynamically as needed,  //
// removing the restrictions imposed by the static nature of native  //
// C strings.                                                        //
// ----------------------------------------------------------------- //

// Constructor and destructor
PUBLIC String    *String_new         (const char *string);
PUBLIC String    *String_copy        (const String *string);
PUBLIC void       String_delete      (String *self);

// Public methods
PUBLIC size_t      String_size       (const String *self);
PUBLIC const char *String_get        (const String *self);
PUBLIC char        String_at         (const String *self, const size_t index);
PUBLIC bool        String_compare    (const String *self, const char *string);

PUBLIC String     *String_set        (String *self, const char *string);
PUBLIC String     *String_set_int    (String *self, char *format, const long int value);
PUBLIC String     *String_set_delim  (String *self, const char *string, const char delimiter, const bool first, const bool until);
PUBLIC String     *String_append     (String *self, const char *string);
PUBLIC String     *String_append_int (String *self, char *format, const long int value);
PUBLIC String     *String_append_flt (String *self, char *format, const double value);
PUBLIC String     *String_prepend    (String *self, const char *string);

PUBLIC String     *String_clear      (String *self);
PUBLIC String     *String_trim       (String *self);

#endif
