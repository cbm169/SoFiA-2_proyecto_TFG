/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (Array_SFX.c) - Source Finding Application               ///
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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "Array_SFX.h"



// ----------------------------------------------------------------- //
// Declaration of private properties and methods of Array_SFX        //
// ----------------------------------------------------------------- //

CLASS Array_SFX
{
	size_t size;
	DATA_T *values;
};



// ----------------------------------------------------------------- //
// Standard constructor                                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) size - Size of the array to be created.                     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to newly created Array_SFX object.                      //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Standard constructor. Will create a new Array_SFX object of     //
//   given size and type and return a pointer to the newly created   //
//   object. Sufficient memory will be allocated to store the array  //
//   values of the specified type. Note that the destructor will     //
//   need to be called explicitly once the object is no longer re-   //
//   quired to release any memory allocated during the lifetime of   //
//   the object. NOTE that the array will be initialised to 0.       //
// ----------------------------------------------------------------- //

PUBLIC Array_SFX *Array_SFX_new(const size_t size)
{
	Array_SFX *self = (Array_SFX *)memory(MALLOC, 1, sizeof(Array_SFX));
	
	self->size = size;
	
	if(self->size) self->values = (DATA_T *)memory(CALLOC, size, sizeof(DATA_T));
	else self->values = NULL;
	
	return self;
}



// ----------------------------------------------------------------- //
// Alternative constructor                                           //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) string - String containing the values to be stored in the   //
//                array, separated by commas.                        //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to newly created Array_SFX object.                      //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Alternative constructor. Will create a new Array_SFX object,    //
//   the size of which is determined by the number of comma-separa-  //
//   ted values specified in 'string'. A pointer to the newly crea-  //
//   ted object will be returned. Sufficient memory will be alloca-  //
//   ted to store the array values. Note that the destructor will    //
//   need to be called explicitly once the object is no longer re-   //
//   quired to release any memory allocated during the lifetime of   //
//   the object.                                                     //
// ----------------------------------------------------------------- //

PUBLIC Array_SFX *Array_SFX_new_str(const char *string)
{
	// Sanity checks
	check_null(string);
	
	// Return empty array if string is empty
	if(!strlen(string)) return Array_SFX_new(0);
	
	// Create a copy of the string
	char *copy = (char *)memory(MALLOC, strlen(string) + 1, sizeof(char));
	strcpy(copy, string);
	
	// Count number of commas
	size_t size = 1;
	size_t i = strlen(copy);
	while(i--) if(copy[i] == ',') ++size;
	
	// Create array of given size
	Array_SFX *self = Array_SFX_new(size);
	
	// Fill array with values
	char *token = strtok(copy, ",");
	ensure(token != NULL, "Failed to parse string as array.");
	
	self->values[0] = (DATA_T)strtod(token, NULL);
	
	for(i = 1; i < size; ++i)
	{
		token = strtok(NULL, ",");
		ensure(token != NULL, "Failed to parse string as array.");
		
		self->values[i] = (DATA_T)strtod(token, NULL);
	}
	
	// Delete string copy again
	free(copy);
	
	return self;
}



// ----------------------------------------------------------------- //
// Copy constructor                                                  //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) source - Array to be copied.                                //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to newly created copy of array object.                  //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Copy constructor. Will create a new array of the same size as   //
//   the source array and then copy all elements from source. A      //
//   pointer to the newly created array copy will be returned. Note  //
//   that the destructor will need to be called explicitly once the  //
//   object is no longer required to release any memory allocated    //
//   during the lifetime the object.                                 //
// ----------------------------------------------------------------- //

PUBLIC Array_SFX *Array_SFX_copy(const Array_SFX *source)
{
	// Sanity checks
	check_null(source);
	
	// Create new array of same size as source
	Array_SFX *self = Array_SFX_new(source->size);
	
	// Copy all elements
	for(size_t i = self->size; i--;) self->values[i] = source->values[i];
	
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
//   mory occupied by the object.                                    //
// ----------------------------------------------------------------- //

PUBLIC void Array_SFX_delete(Array_SFX *self)
{
	if(self != NULL) free(self->values);
	free(self);
	return;
}



// ----------------------------------------------------------------- //
// Get size of array                                                 //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Size of the array.                                              //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning the size of the specified array.    //
// ----------------------------------------------------------------- //

PUBLIC size_t Array_SFX_get_size(const Array_SFX *self)
{
	check_null(self);
	return self->size;
}



// ----------------------------------------------------------------- //
// Get pointer to data array                                         //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to the first element of the array. If the array has     //
//   size 0, a NULL pointer will be returned instead,                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning a pointer to the first element of   //
//   the array.                                                      //
// ----------------------------------------------------------------- //

PUBLIC const DATA_T *Array_SFX_get_ptr(const Array_SFX *self)
{
	check_null(self);
	return self->size ? self->values : NULL;
}



// ----------------------------------------------------------------- //
// Push new element                                                  //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) value    - Value to be added.                               //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to modified array.                                      //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for adding a new element at the end of the array. //
// ----------------------------------------------------------------- //

PUBLIC Array_SFX *Array_SFX_push(Array_SFX *self, const DATA_T value)
{
	check_null(self);
	self->values = (DATA_T *)memory_realloc(self->values, ++self->size, sizeof(DATA_T));
	self->values[self->size - 1] = value;
	return self;
}



// ----------------------------------------------------------------- //
// Get array element                                                 //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the element to be returned.             //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Value of the requested element.                                 //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for retrieving the array value at the specified   //
//   index.                                                          //
// ----------------------------------------------------------------- //

PUBLIC DATA_T Array_SFX_get(const Array_SFX *self, const size_t index)
{
	check_null(self);
	ensure(index < self->size, "Array index out of range.");
	return self->values[index];
}



// ----------------------------------------------------------------- //
// Set array element                                                 //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the element to be set.                  //
//   (3) value    - Value of the element to be set.                  //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to modified array.                                      //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting the value of the array element at the //
//   specified index.                                                //
// ----------------------------------------------------------------- //

PUBLIC Array_SFX *Array_SFX_set(Array_SFX *self, const size_t index, const DATA_T value)
{
	check_null(self);
	ensure(index < self->size, "Array index out of range.");
	self->values[index] = value;
	return self;
}



// ----------------------------------------------------------------- //
// Add value to array element                                        //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the element to be set.                  //
//   (3) value    - Value to be added to the element.                //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to modified array.                                      //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for adding the specified value to the array ele-  //
//   ment at the specified index.                                    //
// ----------------------------------------------------------------- //

PUBLIC Array_SFX *Array_SFX_add(Array_SFX *self, const size_t index, const DATA_T value)
{
	check_null(self);
	ensure(index < self->size, "Array index out of range.");
	self->values[index] += value;
	return self;
}



// ----------------------------------------------------------------- //
// Concatenate two arrays                                            //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) source   - Array to be added.                               //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to modified array.                                      //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for concatenating two arrays by adding the ele-   //
//   ments of source at the end of self.                             //
// ----------------------------------------------------------------- //

PUBLIC Array_SFX *Array_SFX_cat(Array_SFX *self, const Array_SFX *source)
{
	check_null(self);
	if(source == NULL) return self;
	
	self->values = (DATA_T *)memory_realloc(self->values, self->size + source->size, sizeof(DATA_T));
	for(size_t i = 0; i < source->size; ++i) self->values[self->size + i] = source->values[i];
	self->size += source->size;
	
	return self;
}
