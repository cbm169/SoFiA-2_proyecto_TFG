/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (Path.c) - Source Finding Application                    ///
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

#include "Path.h"
#include "String.h"



// ----------------------------------------------------------------- //
// Declaration of private properties and methods of class Path       //
// ----------------------------------------------------------------- //

CLASS Path
{
	String *dir;
	String *file;
	String *path;
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
//   Pointer to newly created Path object.                           //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Standard constructor. Will create a new and empty Path object   //
//   and return a pointer to the newly created object. No memory     //
//   will be allocated other than for the object itself. Note that   //
//   the destructor will need to be called explicitly once the       //
//   object is no longer required to release any memory allocated    //
//   during the lifetime of the object.                              //
// ----------------------------------------------------------------- //

PUBLIC Path *Path_new(void)
{
	Path *self = (Path *)memory(MALLOC, 1, sizeof(Path));
	
	self->dir  = String_new("");
	self->file = String_new("");
	self->path = String_new("");
	
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

PUBLIC void Path_delete(Path *self)
{
	if(self != NULL)
	{
		String_delete(self->dir);
		String_delete(self->file);
		String_delete(self->path);
		free(self);
	}
	
	return;
}



// ----------------------------------------------------------------- //
// Set path from string                                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) path     - String from which to extract path.               //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting the path from the specified string.   //
//   The path can contain just a directory, just a file, or a combi- //
//   nation of both. Anything before the final '/' will be treated   //
//   as a directory and anything after the final '/' as a file name. //
// ----------------------------------------------------------------- //

PUBLIC void Path_set(Path *self, const char *path)
{
	// Sanity checks
	check_null(self);
	check_null(path);
	
	const size_t size = strlen(path);
	ensure(size, "Empty path name encountered.");
	
	// Check for last slash
	const char *delimiter = strrchr(path, '/');
	
	if(delimiter == NULL)
	{
		// No directory, just file name
		String_clear(self->dir);
		String_set(self->file, path);
	}
	else if(delimiter == path + size - 1)
	{
		// No file name, just directory
		String_set(self->dir, path);
		String_clear(self->file);
	}
	else
	{
		// Both directory and file name
		String_set_delim(self->dir, path, '/', false, true);
		String_append(self->dir, "/");
		String_set_delim(self->file, path, '/', false, false);
	}
	
	return;
}



// ----------------------------------------------------------------- //
// Set file name of path                                             //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) file     - String from which to extract file name.          //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting the file name of the specified path.  //
//   The directory name of the path will be left unchanged.          //
// ----------------------------------------------------------------- //

PUBLIC void Path_set_file(Path *self, const char *file)
{
	// Sanity checks
	check_null(self);
	check_null(file);
	
	const size_t size = strlen(file);
	ensure(size, "Empty file name encountered.");
	
	// (Re-)allocate memory and copy file name
	String_set(self->file, file);
	
	return;
}



// ----------------------------------------------------------------- //
// Set directory name of path                                        //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) file     - String from which to extract directory name.     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for setting the directory name of the specified   //
//   path. The file name of the path will be left unchanged.         //
// ----------------------------------------------------------------- //

PUBLIC void Path_set_dir(Path *self, const char *dir)
{
	// Sanity checks
	check_null(self);
	check_null(dir);
	
	const size_t size = strlen(dir);
	ensure(size, "Empty directory name encountered.");
	
	// Copy directory name with trailing slash
	String_set(self->dir, dir);
	if(dir[size - 1] != '/') String_append(self->dir, "/");
	
	return;
}



// ----------------------------------------------------------------- //
// Append directory name of path from template                       //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) basename - Basename to be used for the subdirectory name to //
//                  be appended.                                     //
//   (3) appendix - Suffix to be appended to the subdirectory name.  //
//                  NOTE that any connecting character such as '_'   //
//                  must be explicitly included.                     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for appending a subdirectory to the given direc-  //
//   tory. The name of the subdirectory will be constructed from the //
//   specified basename and appendix. Basename is expected to be a   //
//   file name, and if it includes a mime type suffix (e.g. .fits),  //
//   then that suffix will be removed first before concatenating the //
//   appendix. Note that neither basename nor appendix must contain  //
//   a slash (/).                                                    //
//                                                                   //
//   Example:                                                        //
//                                                                   //
//     Assume self->dir = "/home/user/"                              //
//            basename  = "data.fits"                                //
//            appendix  = "_cubelets"                                //
//                                                                   //
//     Then   self->dir = "/home/user/data_cubelets/"                //
// ----------------------------------------------------------------- //

PUBLIC void Path_append_dir_from_template(Path *self, const char *basename, const char *appendix)
{
	// Sanity checks
	check_null(self);
	check_null(basename);
	check_null(appendix);
	ensure(strchr(basename, '/') == NULL && strchr(appendix, '/') == NULL, "Basename and appendix must not contain \'/\'.");
	
	String *suffix = String_new("");
	String_set_delim(suffix, basename, '.', false, true);
	String_append(suffix, appendix);
	String_append(suffix, "/");
	String_append(self->dir, String_get(suffix));
	String_delete(suffix);
	
	return;
}



// ----------------------------------------------------------------- //
// Construct file name of path from specified template               //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//   (2) basename - Base name to be used for the file name.          //
//   (3) suffix   - Suffix to be appended to basename such that the  //
//                  file name becomes "basename" "suffix". NOTE that //
//                  any connecting character (like '_') must be ex-  //
//                  plicitly included.                               //
//   (4) mimetype - Mime type to be appended to file name such that  //
//                  the name becomes "basename" "suffix" "mimetype". //
//                  NOTE that the dot ('.') must be explicitly in-   //
//                  cluded.                                          //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for constructing the file name of the specified   //
//   path from the given template. The final file name will become   //
//   "basename" "suffix" "mimetype". The directory name of the path  //
//   will remain unchanged.                                          //
//   Note that if 'basename' already contains a mimetype (identified //
//   by the presence of a dot that is not the first character), then //
//   that mimetype ending will be removed first.                     //
//                                                                   //
//   Example:                                                        //
//                                                                   //
//     Assume basename = "data.fits"                                 //
//            suffix   = "-mask"                                     //
//            mimetype = ".fits"                                     //
//                                                                   //
//     Then   filename = "data-mask.fits"                            //
// ----------------------------------------------------------------- //

PUBLIC void Path_set_file_from_template(Path *self, const char *basename, const char *suffix, const char *mimetype)
{
	// Sanity checks
	check_null(self);
	check_null(basename);
	check_null(suffix);
	check_null(mimetype);
	
	String_set_delim(self->file, basename, '.', false, true);
	String_append(self->file, suffix);
	String_append(self->file, mimetype);
	
	return;
}



// ----------------------------------------------------------------- //
// Return the full path as a string                                  //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   String containing the full path.                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for returning the full path of the specified Path //
//   object as a string. The string will be internally stored as     //
//   part of the object and will be automatically deleted whenever   //
//   the destructor is called.                                       //
// ----------------------------------------------------------------- //

PUBLIC const char *Path_get(Path *self)
{
	// Sanity checks
	check_null(self);
	
	String_set(self->path, String_get(self->dir));
	String_append(self->path, String_get(self->file));
	
	return String_get(self->path);
}

// Same, but for directory part only

PUBLIC const char *Path_get_dir(const Path *self)
{
	check_null(self);
	return String_get(self->dir);
}

// Same, but for file part only

PUBLIC const char *Path_get_file(const Path *self)
{
	check_null(self);
	return String_get(self->file);
}



// ----------------------------------------------------------------- //
// Check if file stored in path is readable                          //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self     - Object self-reference.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   True if path is readable, false otherwise.                      //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for testing of the file pointed to by the speci-  //
//   fied path is readable or not. Note that this does not test if   //
//   the file exists, as it could exist but not be readable, e.g. as //
//   a result of insufficient read permission.                       //
// ----------------------------------------------------------------- //

PUBLIC bool Path_file_is_readable(Path *self)
{
	check_null(self);
	FILE *fp = fopen(Path_get(self), "r");
	if(fp == NULL) return false;
	fclose(fp);
	return true;
}
