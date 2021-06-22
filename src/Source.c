/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (Source.c) - Source Finding Application                  ///
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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "Source.h"
#include "String.h"



// ----------------------------------------------------------------- //
// Declaration of private properties and methods of class Source     //
// ----------------------------------------------------------------- //

typedef union SourceValue SourceValue;

union SourceValue
{
	double value_flt;
	size_t value_int;
};

CLASS Source
{
	// Properties
	String         *identifier;
	size_t          n_par;
	SourceValue    *values;
	unsigned char  *types;
	String        **names;
	String        **units;
	String        **ucds;
	int             verbosity;
};

PRIVATE void Source_append_memory(Source *self);



// ----------------------------------------------------------------- //
// Standard constructor                                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   No arguments.                                                   //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to newly created Source object.                         //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Standard constructor. Will create a new and empty Source object //
//   and return a pointer to the newly created object. No memory     //
//   will be allocated other than for the object itself. Note that   //
//   the destructor will need to be called explicitly once the ob-   //
//   ject is no longer required to release any memory allocated du-  //
//   ring the lifetime of the object.                                //
// ----------------------------------------------------------------- //

PUBLIC Source *Source_new(const bool verbosity)
{
	// Allocate memory for new source
	Source *self = (Source *)memory(MALLOC, 1, sizeof(Source));
	
	// Initialise properties
	self->identifier = String_new("");
	self->n_par      = 0;
	self->values     = NULL;
	self->types      = NULL;
	self->names      = NULL;
	self->units      = NULL;
	self->ucds       = NULL;
	
	self->verbosity  = verbosity;
	
	return self;
}



// ----------------------------------------------------------------- //
// Destructor                                                        //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Destructor. Note that the destructor must be called explicitly  //
//   if the object is no longer required. This will release the me-  //
//   mory occupied by the object. NOTE that if the source is added   //
//   to a catalogue object, the destructor MUST NOT be called manu-  //
//   ally, as the catalogue object will take ownership of the source //
//   and call its destructor at the end of its lifetime.             //
// ----------------------------------------------------------------- //

PUBLIC void Source_delete(Source *self)
{
	if(self != NULL)
	{
		for(size_t i = self->n_par; i--;)
		{
			if(self->names != NULL) String_delete(self->names[i]);
			if(self->units != NULL) String_delete(self->units[i]);
			if(self->ucds  != NULL) String_delete(self->ucds[i]);
		}
		
		String_delete(self->identifier);
		
		free(self->values);
		free(self->types);
		free(self->names);
		free(self->units);
		free(self->ucds);
		
		free(self);
	}
	
	return;
}



// ----------------------------------------------------------------- //
// Set source identifier                                             //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) name     - Name to be used as identifier.                   //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting the identifier of a source. Note that //
//   names are case-sensitive.                                       //
// ----------------------------------------------------------------- //

PUBLIC void Source_set_identifier(Source *self, const char *name)
{
	check_null(self);
	String_set(self->identifier, name);
	return;
}



// ----------------------------------------------------------------- //
// Append a new parameter of floating-point type                     //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) name     - Name of the new parameter.                       //
//   (3) value    - Value of the new parameter.                      //
//   (3) unit     - Unit of the new parameter.                       //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for appending a new parameter of floating-point   //
//   type to an existing source. The value will be stored as a       //
//   double-precision floating-point number. NOTE that the function  //
//   does not check if the specified parameter name already exists   //
//   in the current parameter list; if it did, a new parameter with  //
//   the same name would be appended at the end. Note that names are //
//   case-sensitive.                                                 //
// ----------------------------------------------------------------- //

PUBLIC void Source_add_par_flt(Source *self, const char *name, const double value, const char *unit, const char *ucd)
{
	// Sanity checks
	check_null(self);
	check_null(name);
	check_null(unit);
	check_null(ucd);
	
	// Reserve memory for one additional parameter
	Source_append_memory(self);
	
	// Copy new parameter information
	self->values[self->n_par - 1].value_flt = value;
	self->types[self->n_par - 1] = SOURCE_TYPE_FLT;
	String_set(self->names[self->n_par - 1], name);
	String_set(self->units[self->n_par - 1], unit);
	String_set(self->ucds [self->n_par - 1],  ucd);
	
	return;
}



// ----------------------------------------------------------------- //
// Append a new parameter of integer type                            //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) name     - Name of the new parameter.                       //
//   (3) value    - Value of the new parameter.                      //
//   (3) unit     - Unit of the new parameter.                       //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for appending a new parameter of integer type to  //
//   an existing source. The value will be stored as a long inte-    //
//   ger number. NOTE that the function does not check if the speci- //
//   fied parameter name already exists in the current parameter     //
//   list; if it did, a new parameter with the same name would be    //
//   appended at the end. Note that names are case-sensitive.        //
// ----------------------------------------------------------------- //

PUBLIC void Source_add_par_int(Source *self, const char *name, const long int value, const char *unit, const char *ucd)
{
	// Sanity checks
	check_null(self);
	check_null(name);
	check_null(unit);
	check_null(ucd);
	
	// Reserve memory for one additional parameter
	Source_append_memory(self);
	
	// Copy new parameter information
	self->values[self->n_par - 1].value_int = value;
	self->types[self->n_par - 1] = SOURCE_TYPE_INT;
	String_set(self->names[self->n_par - 1], name);
	String_set(self->units[self->n_par - 1], unit);
	String_set(self->ucds [self->n_par - 1],  ucd);
	
	return;
}



// ----------------------------------------------------------------- //
// Set source parameter as floating-point value                      //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) name     - Name of the parameter to be set.                 //
//   (3) value    - Value of the parameter to be set.                //
//   (4) unit     - Unit of the parameter to be set.                 //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting the specified parameter of the speci- //
//   fied source as a double-precision floating-point number. If a   //
//   parameter of the same name already exists, its value will be    //
//   replaced; otherwise, a new parameter will be created and added  //
//   to the existing parameter list. Note that names are case-sensi- //
//   tive.                                                           //
// ----------------------------------------------------------------- //

PUBLIC void Source_set_par_flt(Source *self, const char *name, const double value, const char *unit, const char *ucd)
{
	// Sanity checks
	check_null(self);
	check_null(name);
	check_null(unit);
	check_null(ucd);
	
	// Check if parameter of same name already exists
	size_t index = 0;
	
	if(Source_par_exists(self, name, &index))
	{
		// If so, overwrite with new parameter information
		self->values[index].value_flt = value;
		self->types[index] = SOURCE_TYPE_FLT;
		String_set(self->names[index], name);
		String_set(self->units[index], unit);
		String_set(self->ucds [index],  ucd);
	}
	else
	{
		// Otherwise add as new parameter
		Source_add_par_flt(self, name, value, unit, ucd);
	}
	
	return;
}



// ----------------------------------------------------------------- //
// Set source parameter as integer value                             //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) name     - Name of the parameter to be set.                 //
//   (3) value    - Value of the parameter to be set.                //
//   (4) unit     - Unit of the parameter to be set.                 //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting the specified parameter of the speci- //
//   fied source as a long integer number. If a parameter of the     //
//   same name already exists, its value will be replaced; other-    //
//   wise, a new parameter will be created and added to the existing //
//   parameter list. Note that names are case-sensitive.             //
// ----------------------------------------------------------------- //

PUBLIC void Source_set_par_int(Source *self, const char *name, const long int value, const char *unit, const char *ucd)
{
	// Sanity checks
	check_null(self);
	check_null(name);
	check_null(unit);
	check_null(ucd);
	
	// Check if parameter already exists
	size_t index = 0;
	
	if(Source_par_exists(self, name, &index))
	{
		// If so, overwrite with new parameter information
		self->values[index].value_int = value;
		self->types[index] = SOURCE_TYPE_INT;
		String_set(self->names[index], name);
		String_set(self->units[index], unit);
		String_set(self->ucds [index],  ucd);
	}
	else
	{
		// Otherwise add as new parameter
		Source_add_par_int(self, name, value, unit, ucd);
	}
	
	return;
}



// ----------------------------------------------------------------- //
// Extract a parameter as floating-point value                       //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the parameter to be extracted.          //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Parameter value as type double.                                 //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for extracting the value of the specified parame- //
//   ter from the specified source as a double-precision floating-   //
//   point number.                                                   //
// ----------------------------------------------------------------- //

PUBLIC double Source_get_par_flt(const Source *self, const size_t index)
{
	check_null(self);
	ensure(index < self->n_par, "Source parameter index out of range.");
	return self->values[index].value_flt;
}



// ----------------------------------------------------------------- //
// Extract a parameter as integer value                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the parameter to be extracted.          //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Parameter value as type long int.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for extracting the value of the specified parame- //
//   ter from the specified source as a long integer number.         //
// ----------------------------------------------------------------- //

PUBLIC long int Source_get_par_int(const Source *self, const size_t index)
{
	check_null(self);
	ensure(index < self->n_par, "Source parameter index out of range.");
	return self->values[index].value_int;
}



// ----------------------------------------------------------------- //
// Extract a parameter by name as floating-point value               //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) name     - Name of the parameter to be extracted.           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Parameter value as type double.                                 //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for extracting the value of the specified parame- //
//   ter from the specified source as a double-precision floating-   //
//   point number. If a parameter of the same name does not exist, a //
//   value of NaN will instead be returned.                          //
// ----------------------------------------------------------------- //

PUBLIC double Source_get_par_by_name_flt(const Source *self, const char *name)
{
	check_null(self);
	check_null(name);
	for(size_t i = self->n_par; i--;) if(String_compare(self->names[i], name)) return self->values[i].value_flt;
	warning_verb(self->verbosity, "Parameter \'%s\' not found.", name);
	return NAN;
}



// ----------------------------------------------------------------- //
// Extract a parameter by name as integer value                      //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) name     - Name of the parameter to be extracted.           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Parameter value as type long int.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for extracting the value of the specified parame- //
//   ter from the specified source as a signed long integer value.   //
//   If a parameter of the same name does not exist, a value of 0    //
//   will instead be returned.                                       //
// ----------------------------------------------------------------- //

PUBLIC long int Source_get_par_by_name_int(const Source *self, const char *name)
{
	check_null(self);
	check_null(name);
	for(size_t i = self->n_par; i--;) if(String_compare(self->names[i], name)) return self->values[i].value_int;
	warning_verb(self->verbosity, "Parameter \'%s\' not found.", name);
	return 0;
}



// ----------------------------------------------------------------- //
// Add a position offset to x, y, z parameters                       //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) dx       - Position offset in x.                            //
//   (3) dy       - Position offset in y.                            //
//   (4) dz       - Position offset in z.                            //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for adding a position offset to parameters named  //
//   x, y, z, x_min, x_max, y_min, y_max, z_min, and z_max. Only     //
//   existing parameters will be shifted and non-existing ones ig-   //
//   nored. Offsets can only be positive, as negative pixel coordi-  //
//   nates are not possible.                                         //
// ----------------------------------------------------------------- //

PUBLIC void Source_offset_xyz(Source *self, const size_t dx, const size_t dy, const size_t dz)
{
	check_null(self);
	
	size_t index = 0;
	
	if(Source_par_exists(self, "x", &index)) self->values[index].value_flt += (double)dx;
	if(Source_par_exists(self, "y", &index)) self->values[index].value_flt += (double)dy;
	if(Source_par_exists(self, "z", &index)) self->values[index].value_flt += (double)dz;
	
	if(Source_par_exists(self, "x_min", &index)) self->values[index].value_int += dx;
	if(Source_par_exists(self, "x_max", &index)) self->values[index].value_int += dx;
	if(Source_par_exists(self, "y_min", &index)) self->values[index].value_int += dy;
	if(Source_par_exists(self, "y_max", &index)) self->values[index].value_int += dy;
	if(Source_par_exists(self, "z_min", &index)) self->values[index].value_int += dz;
	if(Source_par_exists(self, "z_max", &index)) self->values[index].value_int += dz;
	
	return;
}



// ----------------------------------------------------------------- //
// Check if source parameter exists                                  //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) name     - Name of the parameter to be checked.             //
//   (3) index    - Pointer to a variable that will hold the index   //
//                  of the source parameter if found.                //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Returns true if the parameter exists, false otherwise.          //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for checking if a parameter of the specified name //
//   already exists in the specified source. The function will re-   //
//   turn true if the parameter exists and false otherwise. Note     //
//   that name is case-sensitive. The variable 'index' will be set   //
//   to the index of the parameter if found. Otherwise it will be    //
//   left untouched. If no index is required, a NULL pointer can in- //
//   stead be provided.                                              //
// ----------------------------------------------------------------- //

PUBLIC bool Source_par_exists(const Source *self, const char *name, size_t *index)
{
	// Sanity checks
	check_null(self);
	check_null(name);
	
	for(size_t i = self->n_par; --i;)
	{
		if(String_compare(self->names[i], name))
		{
			if(index != NULL) *index = i;
			return true;
		}
	}
	
	return false;
}



// ----------------------------------------------------------------- //
// Extract name of parameter by index                                //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the parameter the unit of which is to   //
//                  be returned.                                     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to the name string of the specified parameter.          //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning a pointer to the name string of the //
//   specified parameter.                                            //
// ----------------------------------------------------------------- //

PUBLIC const char *Source_get_name(const Source *self, const size_t index)
{
	check_null(self);
	ensure(index < self->n_par, "Source parameter index out of range.");
	return String_get(self->names[index]);
}



// ----------------------------------------------------------------- //
// Extract unit of parameter by index                                //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the parameter the unit of which is to   //
//                  be returned.                                     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to the unit string of the specified parameter.          //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning a pointer to the unit string of the //
//   specified parameter.                                            //
// ----------------------------------------------------------------- //

PUBLIC const char *Source_get_unit(const Source *self, const size_t index)
{
	check_null(self);
	ensure(index < self->n_par, "Source parameter index out of range.");
	return String_get(self->units[index]);
}



// ----------------------------------------------------------------- //
// Extract type of parameter by index                                //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the parameter the type of which is to   //
//                  be returned.                                     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Data type of the specified parameter.                           //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning the data type of the specified      //
//   parameter, where 0 means integer and 1 means floating point.    //
// ----------------------------------------------------------------- //

PUBLIC unsigned char Source_get_type(const Source *self, const size_t index)
{
	check_null(self);
	ensure(index < self->n_par, "Source parameter index out of range.");
	return self->types[index];
}



// ----------------------------------------------------------------- //
// Extract UCD of parameter by index                                 //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the parameter the UCD of which is to    //
//                  be returned.                                     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to the UCD string of the specified parameter.           //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning a pointer to the Unified Content    //
//   Descriptor (UCD) string of the specified parameter.             //
// ----------------------------------------------------------------- //

PUBLIC const char *Source_get_ucd(const Source *self, const size_t index)
{
	check_null(self);
	ensure(index < self->n_par, "Source parameter index out of range.");
	return String_get(self->ucds[index]);
}



// ----------------------------------------------------------------- //
// Get identifier of specified source                                //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Identifier of the specified source.                             //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning the identifier string of the speci- //
//   fied source. If no identifier has been set, an empty string     //
//   will be returned instead.                                       //
// ----------------------------------------------------------------- //

PUBLIC const char *Source_get_identifier(const Source *self)
{
	check_null(self);
	return String_get(self->identifier);
}



// ----------------------------------------------------------------- //
// Get number of parameters for specified source                     //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Number of parameters currently defined.                         //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning the number of parameters currently  //
//   defined for the specified source.                               //
// ----------------------------------------------------------------- //

PUBLIC size_t Source_get_num_par(const Source *self)
{
	check_null(self);
	return self->n_par;
}



// ----------------------------------------------------------------- //
// Reallocate memory for one additional parameter                    //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Private method for allocating additional memory for one more    //
//   parameter in the specified source. Note that this will not cre- //
//   ate a new parameter yet, but just allocate the memory needed to //
//   append a parameter at the end of the parameter list. The func-  //
//   tion should be called from public member functions that will    //
//   add parameters to a source prior to assigning the new parameter //
//   values.                                                         //
// ----------------------------------------------------------------- //

PRIVATE void Source_append_memory(Source *self)
{
	self->n_par += 1;
	self->values = (SourceValue *)   memory_realloc(self->values, self->n_par, sizeof(SourceValue));
	self->types  = (unsigned char *) memory_realloc(self->types,  self->n_par, sizeof(unsigned char));
	self->names  = (String **)       memory_realloc(self->names,  self->n_par, sizeof(String *));
	self->units  = (String **)       memory_realloc(self->units,  self->n_par, sizeof(String *));
	self->ucds   = (String **)       memory_realloc(self->ucds,   self->n_par, sizeof(String *));
	
	// Call constructor on all strings
	self->names[self->n_par - 1] = String_new("");
	self->units[self->n_par - 1] = String_new("");
	self->ucds [self->n_par - 1] = String_new("");
	
	return;
}
