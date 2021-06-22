/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (Map.c) - Source Finding Application                     ///
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
#include "Map.h"



// ----------------------------------------------------------------- //
// Declaration of private properties and methods of class Map        //
// ----------------------------------------------------------------- //

CLASS Map
{
	size_t  size;
	size_t *keys;
	size_t *values;
};



// ----------------------------------------------------------------- //
// Standard constructor                                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   No arguments.                                                   //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to newly created Map object.                            //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Standard constructor. Will create a new, empty Map object and   //
//   return a pointer to the newly created object. Note that the de- //
//   structor will need to be called explicitly once the object is   //
//   no longer required to release any memory allocated during the   //
//   lifetime of the object.                                         //
// ----------------------------------------------------------------- //

PUBLIC Map *Map_new(void)
{
	Map *self = (Map *)memory(MALLOC, 1, sizeof(Map));
	
	self->size = 0;
	self->keys = NULL;
	self->values = NULL;
	
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

PUBLIC void Map_delete(Map *self)
{
	if(self != NULL)
	{
		free(self->keys);
		free(self->values);
		free(self);
	}
	
	return;
}



// ----------------------------------------------------------------- //
// Push new key-value pair onto map                                  //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) key      - Key to be created.                               //
//   (3) value    - Value to be added.                               //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for pushing a new key-value pair onto the speci-  //
//   fied map. Note that there will be no check as to whether the    //
//   key already exists, and it is therefore possible to create more //
//   than one entry with the same key.                               //
// ----------------------------------------------------------------- //

PUBLIC void Map_push(Map *self, const size_t key, const size_t value)
{
	// Sanity checks
	check_null(self);
	
	++self->size;
	
	self->keys   = (size_t *)memory_realloc(self->keys, self->size, sizeof(size_t));
	self->values = (size_t *)memory_realloc(self->values, self->size, sizeof(size_t));
	
	self->keys[self->size - 1] = key;
	self->values[self->size - 1] = value;
	
	return;
}



// ----------------------------------------------------------------- //
// Retrieve value by key                                             //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) key      - Key to be retrieved.                             //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Value belonging to key.                                         //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for retrieving the value associated with the spe- //
//   cified key. If the same key exists more than once, the last oc- //
//   currence will be retrieved. The process will be terminated of   //
//   the specified map is empty or the key is not found.             //
// ----------------------------------------------------------------- //

PUBLIC size_t Map_get_value(const Map *self, const size_t key)
{
	// Sanity checks
	check_null(self);
	
	// Search for key and return value
	for(size_t i = 0; i < self->size; ++i) if(self->keys[i] == key) return self->values[i];
	
	// Key not found
	ensure(false, "Key \'%zu\' not found in map.", key);
	
	// While this return statement is never executed, it is required by the compiler:
	return 0;
}



// ----------------------------------------------------------------- //
// Return size of map                                                //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Current size of map, i.e. number of entries.                    //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning the current size of the specified   //
//   map. i.e. the number of key-value pairs currently stored. If    //
//   same key exists multiple times, it will also be counted repea-  //
//   tedly.                                                          //
// ----------------------------------------------------------------- //

PUBLIC size_t Map_get_size(const Map *self)
{
	check_null(self);
	return self->size;
}



// ----------------------------------------------------------------- //
// Check if key exists                                               //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) key      - Key to be checked.                               //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   True if key exists, false otherwise.                            //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for checking if the specified key exists. The     //
//   method will return true if the key is found and false other-    //
//   wise.                                                           //
// ----------------------------------------------------------------- //

PUBLIC bool Map_key_exists(const Map *self, const size_t key)
{
	// Sanity checks
	check_null(self);
	
	// Search for key
	for(size_t i = 0; i < self->size; ++i) if(self->keys[i] == key) return true;
	return false;
}
