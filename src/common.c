/// ____________________________________________________________________ ///
///                                                                      ///
/// SoFiA 2.1.1 (common.c) - Source Finding Application                  ///
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
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>

#include "common.h"



// ----------------------------------------------------------------- //
// Terminate programme execution if condition is false               //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) condition - Condition to be tested. If false, programme     //
//                   execution will be terminated with an error      //
//                   message.                                        //
//   (2) format    - Message to be printed. This can contain option- //
//                   al format specifiers as used in the printf()    //
//                   function.                                       //
//   (3) ...       - Optional parameters to be printed as defined by //
//                   the format specifiers, if present.              //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   If the specified condition is false, an error message will be   //
//   printed and execution of the programme will be terminated with  //
//   a signal of 1. The specified error message can contain optional //
//   format specifiers as used in the printf() function, in which    //
//   case additional arguments need to be supplied that will be      //
//   printed as part of the message.                                 //
// ----------------------------------------------------------------- //

void ensure(const bool condition, const char *format, ...)
{
	if(!condition)
	{
		va_list args;
		va_start(args, format);
		fprintf(stderr, "\n\33[31mERROR: ");
		vfprintf(stderr, format, args);
		fprintf(stderr, "\33[0m\n\n");
		va_end(args);
		exit(1);
	}
	return;
}



// ----------------------------------------------------------------- //
// Check if pointer is NULL                                          //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) ptr - Pointer to be checked.                                //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   This function will check if the supplied pointer is NULL and    //
//   terminate the current process in this case. Otherwise, the      //
//   function will do nothing.                                       //
// ----------------------------------------------------------------- //

void check_null(const void *ptr)
{
	ensure(ptr != NULL, "NULL pointer encountered.");
	return;
}



// ----------------------------------------------------------------- //
// Print informational message                                       //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) format - Message to be printed. This can contain optional   //
//                format specifiers as in the printf() function.     //
//   (2) ...    - Optional parameters to be printed as defined by    //
//                the format specifies, if present.                  //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Prints an informational message to standard output. The message //
//   can contain optional format specifiers as used in the printf()  //
//   function, in which case additional arguments need to be sup-    //
//   plied that will be printed as part of the message.              //
// ----------------------------------------------------------------- //

void message(const char *format, ...)
{
	va_list args;
	va_start(args, format);
	printf("  ");
	vprintf(format, args);
	printf("\n");
	va_end(args);
	return;
}

// Same, but with additional verbosity option

void message_verb(const bool verbosity, const char *format, ...)
{
	if(verbosity)
	{
		va_list args;
		va_start(args, format);
		printf("  ");
		vprintf(format, args);
		printf("\n");
		va_end(args);
	}
	
	return;
}



// ----------------------------------------------------------------- //
// Print status message                                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) format - Message to be printed. This can contain optional   //
//                format specifiers as in the printf() function.     //
//   (2) ...    - Optional parameters to be printed as defined by    //
//                the format specifies, if present.                  //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Prints an informational status message to standard output. The  //
//   message will be highlighted in some way in the terminal window  //
//   to distinguish it from standard messages. The message can con-  //
//   tain optional format specifiers as used in the printf() func-   //
//   tion, in which case additional arguments need to be supplied    //
//   that will be printed as part of the message.                    //
// ----------------------------------------------------------------- //

void status(const char *format, ...)
{
	va_list args;
	va_start(args, format);
	printf("\33[36m____________________________________________________________________________\33[0;1m\n\n ");
	vprintf(format, args);
	printf("\n\33[0;36m____________________________________________________________________________\33[0m\n\n");
	va_end(args);
	return;
}



// ----------------------------------------------------------------- //
// Print warning message                                             //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) format - Message to be printed. This can contain optional   //
//                format specifiers as in the printf() function.     //
//   (2) ...    - Optional parameters to be printed as defined by    //
//                the format specifies, if present.                  //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Prints the word 'WARNING' followed by the specified message and //
//   a final newline character to standard error. The message can    //
//   contain format specifiers as used in the printf() function, in  //
//   which case additional arguments are expected that will be prin- //
//   ted as part of the message. Output will be suppressed if the    //
//   VERBOSE parameter is set to false.                              //
// ----------------------------------------------------------------- //

void warning(const char *format, ...)
{
	va_list args;
	va_start(args, format);
	fprintf(stderr, "\33[33mWARNING: ");
	vfprintf(stderr, format, args);
	fprintf(stderr, "\33[0m\n");
	va_end(args);
	
	return;
}

// Same, but with additional verbosity option

void warning_verb(const bool verbosity, const char *format, ...)
{
	if(verbosity)
	{
		va_list args;
		va_start(args, format);
		fprintf(stderr, "\33[33mWARNING: ");
		vfprintf(stderr, format, args);
		fprintf(stderr, "\33[0m\n");
		va_end(args);
	}
	
	return;
}



// ----------------------------------------------------------------- //
// Display progress bar                                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) text     - Text to appear in front of the progress bar.     //
//   (2) progress - Progress value relative to maximum.              //
//   (3) maximum  - Maximum possible progress value.                 //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Displays a progress bar showing the current progress relative   //
//   to the maximum possible progress. The progress bar can be up-   //
//   dated by calling this function repeatedly until progress ==     //
//   maximum, in which case the progress is shown as 100%, and a     //
//   newline character will be printed to end the progress bar up-   //
//   date. No other output must occur in between the first and last  //
//   call to this function, as otherwise the progress bar would be   //
//   broken over onto a new line.                                    //
// ----------------------------------------------------------------- //

void progress_bar(const char *text, const size_t progress, const size_t maximum)
{
	// Sanity checks
	if(!maximum || maximum < progress) return;
	
	const size_t size = 50;
	const size_t cur = size * progress / maximum;
	const bool status = (progress < maximum);
	size_t i;
	
	if(progress < maximum) printf("  %s |\33[33m", text);
	else printf("  %s |\33[32m", text);
	for(i = 0; i < cur; ++i) printf("=");
	if(status) for(i = cur; i < size; ++i) printf(" ");
	printf("\33[0m| %zu%%\r", 100 * progress / maximum);
	if(!status) printf("\n\n");
	fflush(stdout);
	
	return;
}



// ----------------------------------------------------------------- //
// Print time stamp                                                  //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) start       - Start time relative to which the elapsed time //
//                     is calculated.                                //
//   (2) start_clock - Start time in units of clock cycles relative  //
//                     to which the elapsed CPU time is calculated.  //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Prints the elapsed clock time as well as the elapsed CPU time   //
//   as the difference between the current time and the specified    //
//   start time.                                                     //
// ----------------------------------------------------------------- //

void timestamp(const time_t start, const clock_t start_clock)
{
	const unsigned long dt = difftime(time(NULL), start);
	const unsigned long h =  dt / 3600;
	const unsigned long m = (dt - 3600 * h) / 60;
	const unsigned long s =  dt - 3600 * h  - 60 * m;
	
	const unsigned long dt_clock = (unsigned long)((double)(clock() - start_clock) / CLOCKS_PER_SEC);
	const unsigned long h_clock =  dt_clock / 3600;
	const unsigned long m_clock = (dt_clock - 3600 * h_clock) / 60;
	const unsigned long s_clock =  dt_clock - 3600 * h_clock  - 60 * m_clock;

	printf("\n\33[36m  Elapsed time: %02ld:%02ld:%02ld h\n", h, m, s);
	printf("  CPU time:     %02ld:%02ld:%02ld h\33[0m\n", h_clock, m_clock, s_clock);
	return;
}



// ----------------------------------------------------------------- //
// Wrapper around malloc() / calloc()                                //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) mode       - MALLOC for malloc() or CALLOC for calloc().    //
//   (2) n_blocks   - Number of array elements to be stored.         //
//   (3) block_size - Size of each array element.                    //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to allocated memory.                                    //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Wrapper function for malloc() and calloc(). The function will   //
//   reserve memory for n_blocks elements of size block_size and re- //
//   turn a pointer of type void to the allocated memory. If memory  //
//   allocation fails, an error message will be printed and the cur- //
//   rent process terminated.                                        //
// ----------------------------------------------------------------- //

void *memory(const int mode, const size_t n_blocks, const size_t block_size)
{
	ensure(n_blocks && block_size, "Cannot allocate memory block of zero size.");
	void *ptr = (mode == CALLOC) ? calloc(n_blocks, block_size) : malloc(n_blocks * block_size);
	ensure(ptr != NULL, "Failed to allocate %f GB of memory.", (double)(n_blocks * block_size) / GIGABYTE);
	return ptr;
}



// ----------------------------------------------------------------- //
// Wrapper around realloc()                                          //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) ptr        - Pointer to memory address to be reallocated.   //
//   (2) n_blocks   - Number of array elements to be stored.         //
//   (3) block_size - Size of each array element.                    //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Pointer to allocated memory.                                    //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Wrapper function for realloc(). The function will reallocate    //
//   memory for n_blocks elements of size block_size and return a    //
//   pointer of type void to the allocated memory block. If memory   //
//   allocation fails, an error message will be printed and the cur- //
//   rent process terminated.                                        //
// ----------------------------------------------------------------- //

void *memory_realloc(void *ptr, const size_t n_blocks, const size_t block_size)
{
	ensure(n_blocks && block_size, "Cannot reallocate memory block of zero size.");
	void *ptr2 = realloc(ptr, n_blocks * block_size);
	ensure(ptr2 != NULL, "Failed to allocate %f GB of memory.", (double)(n_blocks * block_size) / GIGABYTE);
	return ptr2;
}



// ----------------------------------------------------------------- //
// Trim whitespace from string                                       //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) str - String to be trimmed.                                 //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Returns a pointer to the beginning of the trimmed string.       //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Trims a string by removing whitespace from the beginning and    //
//   end of the string. Whitespace will be any character that is     //
//   considered as space by the isspace() function of the standard   //
//   library.                                                        //
//                                                                   //
//   Note that this will modify the original string and return a     //
//   pointer to a different position in the specified string. This   //
//   means that the original pointer must not be overwritten or dis- //
//   carded as otherwise its memory allocation could not be freed    //
//   any more! Furthermore, the returned string pointer must not be  //
//   freed under any circumstances, as it points to memory already   //
//   allocated elsewhere.                                            //
// ----------------------------------------------------------------- //

char *trim_string(char *str)
{
	if(str != NULL)
	{
		// Trim leading whitespace
		while(isspace((unsigned char)*str)) ++str;
		
		// All whitespace?
		if(*str == 0)  return str;
		
		// Trim trailing whitespace
		char *end = str + strlen(str) - 1;
		while(end > str && isspace((unsigned char)*end)) --end;
		
		// Terminate string with NUL
		end[1] = '\0';
	}
	
	return str;
}



// ----------------------------------------------------------------- //
// Convert integer number to string                                  //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) str   - String to hold the result.                          //
//   (2) size  - Size of the string.                                 //
//   (3) value - Integer value to convert.                           //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Function for converting the specified integer value into a      //
//   string of size 'size' pointed to by 'str'. The user will need   //
//   to ensure that the string is large enough to be able to hold    //
//   the result. If the string is too small, the value will be trun- //
//   cated.                                                          //
// ----------------------------------------------------------------- //

void int_to_str(char *str, const size_t size, const long int value)
{
	snprintf(str, size, "%ld", value);
	return;
}



// ----------------------------------------------------------------- //
// Swap two values                                                   //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   val1  - Pointer to first value.                                 //
//   val2  - Pointer to second value.                                //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Function for swapping two double-precision floating-point       //
//   values.                                                         //
// ----------------------------------------------------------------- //

void swap(double *val1, double *val2)
{
	check_null(val1);
	check_null(val2);
	
	double tmp = *val1;
	*val1 = *val2;
	*val2 = tmp;
	
	return;
}



// ----------------------------------------------------------------- //
// Determine optimal plot tick mark interval                         //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) range - Plot range.                                         //
//   (2) n     - Desired number of tick marks.                       //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Optimal tick mark interval.                                     //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Function for automatically determining the optimal interval     //
//   between tick marks on a plot based on the specified plotting    //
//   range and the desired total number of tick marks. The interval  //
//   will be optimised such that approximately the right number of   //
//   tick marks will be spread across the specified range while      //
//   setting the tick mark intervals to either 1, 2, 5 or 10 times   //
//   10^N, where N is an integer number. Values for n should be in   //
//   the range of about 3-5 for optimal results.                     //
// ----------------------------------------------------------------- //

double auto_tick(const double range, const size_t n)
{
	const double tick  = fabs(range) / n;
	const double tick2 = pow(10.0, floor(log10(tick)));
	
	const double dist1 = fabs(tick / tick2 -  1.0);
	const double dist2 = fabs(tick / tick2 -  2.0);
	const double dist3 = fabs(tick / tick2 -  5.0);
	const double dist4 = fabs(tick / tick2 - 10.0);
	
	if(dist1 < dist2)      return  1.0 * tick2;
	else if(dist2 < dist3) return  2.0 * tick2;
	else if(dist3 < dist4) return  5.0 * tick2;
	else                   return 10.0 * tick2;
}



// ----------------------------------------------------------------- //
// Write header of EPS file                                          //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) fp      - File pointer                                      //
//   (2) title   - Title of the EPS document.                        //
//   (3) creator - Creator of the EPS file.                          //
//   (4) bbox    - Bounding box of the form "xmin ymin xmax ymax".   //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Function for writing header information to an EPS file pointed  //
//   to by fp. The user can specify the title, creator and bounding  //
//   box of the file. In addition, a number of useful procedures     //
//   will be defined.                                                //
// ----------------------------------------------------------------- //

void write_eps_header(FILE *fp, const char *title, const char *creator, const char *bbox)
{
	fprintf(fp, "%%!PS-Adobe-3.0 EPSF-3.0\n");
	fprintf(fp, "%%%%Title: %s\n", title);
	fprintf(fp, "%%%%Creator: %s\n", creator);
	fprintf(fp, "%%%%BoundingBox: %s\n", bbox);
	fprintf(fp, "%%%%EndComments\n");
	fprintf(fp, "/m {moveto} bind def\n");
	fprintf(fp, "/l {lineto} bind def\n");
	fprintf(fp, "/a {arc} bind def\n");
	fprintf(fp, "/af {arc fill} bind def\n");
	fprintf(fp, "/as {arc stroke} bind def\n");
	fprintf(fp, "/s {stroke} bind def\n");
	fprintf(fp, "/f {fill} bind def\n");
	fprintf(fp, "/rgb {setrgbcolor} bind def\n");
	fprintf(fp, "/np {newpath} bind def\n");
	fprintf(fp, "/cp {closepath} bind def\n");
	fprintf(fp, "/lw {setlinewidth} bind def\n");
	fprintf(fp, "/Helvetica findfont 12 scalefont setfont\n");
	fprintf(fp , "/ellipse {gsave /scf exch def /pa exch def /rmin exch def /rmaj exch def /posy exch def /posx exch def 0.5 setlinewidth newpath posx posy translate matrix currentmatrix 1 scf scale pa rotate 1 rmin rmaj div scale 0 0 rmaj 0 360 arc closepath setmatrix stroke grestore} bind def\n");
	return;
}



// ----------------------------------------------------------------- //
// Write footer of EPS file                                          //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) fp      - File pointer                                      //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Function for writing footer information to an EPS file pointed  //
//   to by fp. Combined with print_eps_header() this will create a   //
//   valid EPS file.                                                 //
// ----------------------------------------------------------------- //

void write_eps_footer(FILE *fp)
{
	fprintf(fp, "showpage\n");
	fprintf(fp, "%%%%EndDocument\n");
	return;
}





// ----------------------------------------------------------------- //
// Check native endianness of system                                 //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   No arguments.                                                   //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   Returns true on little-endian and false on big-endian machines. //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Static function for checking the native endianness of the ma-   //
//   chine at run time. Note that the result will only be correct on //
//   machines where sizeof(char) < sizeof(int).                      //
// ----------------------------------------------------------------- //

bool is_little_endian(void)
{
	const unsigned int n = 1U;
	return *((unsigned char *)&n) == 1U;
}



// ----------------------------------------------------------------- //
// Swap the byte order of a multi-byte word                          //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) word - Pointer to a multi-byte value.                       //
//   (2) size - Number of bytes used by word; must be 2, 4 or 8.     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Static function for reversing the order of the bytes in a mul-  //
//   ti-byte word provided as an array of char with given size. Only //
//   word sizes of 2, 4 or 8 are currently supported. Note that this //
//   function will only work correctly on systems where CHAR_BIT is  //
//   equal to 8. In addition, the built-in byte swap functions from  //
//   GCC are required.                                               //
// ----------------------------------------------------------------- //

void swap_byte_order(char *word, const size_t size)
{
	if     (size == 2) *((uint16_t *)word) = __builtin_bswap16(*((uint16_t *)word));
	else if(size == 4) *((uint32_t *)word) = __builtin_bswap32(*((uint32_t *)word));
	else if(size == 8) *((uint64_t *)word) = __builtin_bswap64(*((uint64_t *)word));
	
	return;
}
