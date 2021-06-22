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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "String.h"



// ----------------------------------------------------------------- //
// Declaration of private properties and methods of class String     //
// ----------------------------------------------------------------- //

CLASS String
{
	size_t size;   // String size WITHOUT terminating null character!
	char *string;
};



// ----------------------------------------------------------------- //
// Standard constructor                                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   string - String to be assigned to the new String object. Use "" //
//            to create an empty string.                             //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to newly created String object.                         //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Standard constructor. Will create a new and empty String ob-    //
//   ject. Note that the destructor will have to be called explicit- //
//   ly once the object is no longer required to release its memory  //
//   again.                                                          //
// ----------------------------------------------------------------- //

PUBLIC String *String_new(const char *string)
{
	// Sanity check
	if(string == NULL) return NULL;
	
	// Allocate memory for new String object
	String *self = (String *)memory(MALLOC, 1, sizeof(String));
	
	// Initialise properties
	self->size = 0;
	self->string = NULL;
	
	// Call method for setting string value
	String_set(self, string);
	
	// Return new String object
	return self;
}



// ----------------------------------------------------------------- //
// Copy constructor                                                  //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   string - String object to be copied.                            //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to newly created String object.                         //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Copy constructor. Will create a copy of the specified String    //
//   object. Note that this copy constructor has only been included  //
//   for semantic reasons, as its sole job is to call the standard   //
//   constructor.                                                    //
// ----------------------------------------------------------------- //

PUBLIC String *String_copy(const String *string)
{
	return String_new(string == NULL ? NULL : string->string);
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
//   if the object is no longer required. This will release all me-  //
//   mory occupied by the object.                                    //
// ----------------------------------------------------------------- //

PUBLIC void String_delete(String *self)
{
	if(self != NULL) free(self->string);
	free(self);
	return;
}



// ----------------------------------------------------------------- //
// Return string size                                                //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Size of the string without the terminating null character.      //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning the size of the string excluding    //
//   the terminating null character.                                 //
// ----------------------------------------------------------------- //

PUBLIC size_t String_size(const String *self)
{
	return self == NULL ? 0 : self->size;
}



// ----------------------------------------------------------------- //
// Return pointer to C string                                        //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to C string of self, or NULL is self is NULL.           //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning a pointer to the C string stored in //
//   self. If self is NULL, a NULL pointer will instead be returned. //
// ----------------------------------------------------------------- //

PUBLIC const char *String_get(const String *self)
{
	return self == NULL ? NULL : self->string;
}



// ----------------------------------------------------------------- //
// Extract character at specified index                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) index    - Index of the character to be extracted.          //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Character at the specified index.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning the character at the specified in-  //
//   dex. The process will be terminated if the index is found to be //
//   out of range.                                                   //
// ----------------------------------------------------------------- //

PUBLIC char String_at(const String *self, const size_t index)
{
	// Sanity checks
	check_null(self);
	ensure(index < self->size, "String index out of range.");
	
	return self->string[index];
}



// ----------------------------------------------------------------- //
// Compare two strings                                               //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) string   - String object to compare to.                     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   True if strings are equal, false otherwise.                     //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for checking if two string objects are equal,     //
//   i.e. if they contain the same character sequence. The method    //
//   will return true if the strings are equal and false otherwise.  //
// ----------------------------------------------------------------- //

PUBLIC bool String_compare(const String *self, const char *string)
{
	// Sanity checks
	check_null(self);
	check_null(string);
	
	return strcmp(self->string, string) == 0;
}



// ----------------------------------------------------------------- //
// Set string                                                        //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) string   - String value to be set.                          //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to self after assignment.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting a string to the specified value. The  //
//   string will be cleared if the specified value is an empty       //
//   string or a NULL pointer.                                       //
// ----------------------------------------------------------------- //

PUBLIC String *String_set(String *self, const char *string)
{
	// Sanity checks
	check_null(self);
	
	// Empty string?
	if(string == NULL || *string == '\0')
	{
		String_clear(self);
	}
	else
	{
		self->size = strlen(string);
		self->string = (char *)memory_realloc(self->string, self->size + 1, sizeof(char));
		strcpy(self->string, string);
	}
	
	return self;
}



// ----------------------------------------------------------------- //
// Set string from integer                                           //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) value    - Integer value to set the string to.              //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to self after assignment.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting a string to contain a textual repre-  //
//   sentation of the specified integer value. The size of the re-   //
//   sulting string is restricted to at most 32 characters which     //
//   should be sufficient to hold all possible integer values. A     //
//   warning would be issued if the buffer size was insufficient     //
//   and the string be truncated to 32 characters (including the     //
//   terminating null character) in this case.                       //
// ----------------------------------------------------------------- //

PUBLIC String *String_set_int(String *self, char *format, const long int value)
{
	// Sanity checks
	check_null(self);
	if(format == NULL) format = "%ld";
	
	// Create buffer
	char *buffer = (char *)memory(MALLOC, 32, sizeof(char));
	
	// Write number into buffer
	const int flag = snprintf(buffer, 32, format, value);
	if(flag >= 32) warning("Buffer overflow in int-to-string conversion.");
	if(flag <   0) warning("Encoding error in int-to-string conversion.");
	
	// Copy buffer into string
	String_set(self, buffer);
	
	// Clean up
	free(buffer);
	
	return self;
}



// ----------------------------------------------------------------- //
// Set string from or until delimiting character                     //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self      - Object self-reference.                          //
//   (2) string    - String value to set the String object to.       //
//   (3) delimiter - Delimiting character from or up to which to     //
//                   copy the string value.                          //
//   (4) first     - If true, use the first occurrence of the deli-  //
//                   miter (first from start), otherwise use the     //
//                   last occurrence (last before end).              //
//   (5) until     - If true, copy the string until the delimiting   //
//                   character, otherwise copy the string from the   //
//                   delimiting character onwards. The delimiter it- //
//                   self will be excluded in both cases.            //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to self after assignment.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting a String object to the specified      //
//   string up to or until the specified delimiting character. The   //
//   delimiting character can either be the first from the start or  //
//   the last before the end (argument 'first'), and either the sub- //
//   string until the delimiter or from the delimiter onwards can be //
//   copied (argument 'until'). The delimiting character itself will //
//   be excluded in all cases. This method can be used to copy part  //
//   of a string as defined by a specific delimiting character.      //
// ----------------------------------------------------------------- //

PUBLIC String *String_set_delim(String *self, const char *string, const char delimiter, const bool first, const bool until)
{
	// Sanity checks
	check_null(self);
	check_null(string);
	
	// Some settings
	const size_t size_old = strlen(string);
	size_t size_new = 0;
	char *pos_delim = NULL;
	
	// Find delimiter
	if(first) pos_delim = strchr(string, delimiter);
	else pos_delim = strrchr(string, delimiter);
	
	// Not found?
	if(pos_delim == NULL) return String_set(self, string);
	
	// Determine size of new sub-string
	if(until) size_new = pos_delim - string;
	else size_new = string + size_old - pos_delim - 1;
	
	// If delimiter at edge, clear string
	if(size_new == 0) return String_clear(self);
	
	// Otherwise copy sub-string
	char *tmp = (char *)memory(MALLOC, size_new + 1, sizeof(char));
	if(until) strncpy(tmp, string, size_new);
	else strncpy(tmp, pos_delim + 1, size_new);
	*(tmp + size_new) = '\0';
	
	String_set(self, tmp);
	free(tmp);
	
	return self;
}



// ----------------------------------------------------------------- //
// Append string                                                     //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) string   - String value to be appended.                     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to self after assignment.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for appending a String object with the specified  //
//   string value. The size of the String object will be automati-   //
//   cally adjusted.                                                 //
// ----------------------------------------------------------------- //

PUBLIC String *String_append(String *self, const char *string)
{
	// Sanity checks
	check_null(self);
	if(string == NULL || *string == '\0') return self;
	
	const size_t size = strlen(string);
	
	self->string = (char *)memory_realloc(self->string, self->size + size + 1, sizeof(char));
	strcpy(self->string + self->size, string);
	self->size += size;
	
	return self;
}



// ----------------------------------------------------------------- //
// Append integer value to string                                    //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) value    - Integer value the textual representation of      //
//                  which will be appended to the string.            //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to self after assignment.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for appending the textual representation of the   //
//   specified integer value to a string. The size of the resulting  //
//   string is restricted to at most 32 characters which should be   //
//   sufficient to hold all possible integer values. A warning would //
//   be issued if the buffer size was insufficient and the string be //
//   truncated to 32 characters (including the terminating null      //
//   character) in this case.                                        //
// ----------------------------------------------------------------- //

PUBLIC String *String_append_int(String *self, char *format, const long int value)
{
	// Sanity checks
	check_null(self);
	if(format == NULL) format = "%ld";
	
	// Create buffer
	char *buffer = (char *)memory(MALLOC, 32, sizeof(char));
	
	// Write number into buffer
	const int flag = snprintf(buffer, 32, format, value);
	if(flag >= 32) warning("Buffer overflow in int-to-string conversion.");
	if(flag <   0) warning("Encoding error in int-to-string conversion.");
	
	// Append buffer to string
	String_append(self, buffer);
	
	// Clean up
	free(buffer);
	
	return self;
}



// ----------------------------------------------------------------- //
// Append floating-point value to string                             //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) format   - Format specifier as used by printf(). If set to  //
//                  NULL, the default format will be applied.        //
//   (3) value    - Floating-point value the textual representation  //
//                  of which will be appended to the string.         //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to self after assignment.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for appending the textual representation of the   //
//   specified floating-point value to a string. The format can be   //
//   specified as in the printf() function. If no format is given    //
//   (i.e. format is a NULL pointer) then %.5f will be used by de-   //
//   fault. The resulting string is restricted to 32 characters (in- //
//   cluding the terminating null character) and will be truncated   //
//   with a warning message if exceeded.                             //
// ----------------------------------------------------------------- //

PUBLIC String *String_append_flt(String *self, char *format, const double value)
{
	// Sanity checks
	check_null(self);
	if(format == NULL) format = "%.5f";
	
	// Create buffer
	char *buffer = (char *)memory(MALLOC, 32, sizeof(char));
	
	// Write number into buffer
	const int flag = snprintf(buffer, 32, format, value);
	if(flag >= 32) warning("Buffer overflow in float-to-string conversion.");
	if(flag <   0) warning("Encoding error in float-to-string conversion.");
	
	// Append buffer to string
	String_append(self, buffer);
	
	// Clean up
	free(buffer);
	
	return self;
}



// ----------------------------------------------------------------- //
// Prepend string                                                    //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) format   - String to be prepended.                          //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to self after assignment.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for prepending a string with the specified string //
//   value. The size of the string will be automatically adjusted.   //
// ----------------------------------------------------------------- //

PUBLIC String *String_prepend(String *self, const char *string)
{
	// Sanity checks
	check_null(self);
	if(string == NULL || *string == '\0') return self;
	
	const size_t size = strlen(string);
	
	self->string = (char *)memory_realloc(self->string, self->size + size + 1, sizeof(char));
	memmove(self->string + size, self->string, self->size + 1);
	memcpy(self->string, string, size);
	self->size += size;
	
	return self;
}



// ----------------------------------------------------------------- //
// Clear string                                                      //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to self after assignment.                               //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for clearing the specified String object. This    //
//   will set the string to an empty C string ("\0") and the size of //
//   the string to zero.                                             //
// ----------------------------------------------------------------- //

PUBLIC String *String_clear(String *self)
{
	// Sanity checks
	check_null(self);
	
	self->size = 0;
	self->string = (char *)memory_realloc(self->string, 1, sizeof(char));
	*self->string = '\0';
	
	return self;
}



// ----------------------------------------------------------------- //
// Trim string                                                       //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to self after trimming.                                 //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for trimming a string by removing any contiguous  //
//   sequence of whitespace characters from the beginning and end of //
//   the string. Whitespace will be anything considered as white-    //
//   space by the standard library function isspace().               //
// ----------------------------------------------------------------- //

PUBLIC String *String_trim(String *self)
{
	// Sanity checks
	if(self == NULL || self->size == 0) return self;
	
	// Find first non-whitespace character
	char *start = self->string;
	while(isspace((unsigned char)*start)) ++start;
	
	// All space?
	if(*start == '\0')
	{
		String_clear(self);
		return self;
	}
	
	// Find last non-whitespace character
	char *end = self->string + self->size - 1;
	while(end > start && isspace((unsigned char)*end)) --end;
	
	// Shift sub-string to beginning of string buffer
	self->size = end - start + 1;
	memmove(self->string, start, self->size);
	*(self->string + self->size) = '\0';
	
	// Adjust memory allocation
	self->string = (char *)memory_realloc(self->string, self->size + 1, sizeof(char));
	
	return self;
}
