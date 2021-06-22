
#ifndef DATACUBE_CL_H
#define DATACUBE_CL_H

//#include "OpenCL-Headers-master/CL/cl_version.h"
//#include "OpenCL-Headers-master/CL/opencl.h"
//#include "OpenCL-Headers-master/CL/cl.h"
#include <CL/cl.h>


	#include <stdlib.h>
	#include <stdio.h>
	
	#include <string.h>
	#include <math.h>
	#include <stdint.h>
	#include <limits.h>

	#include "WCS.h"
	#include "DataCube.h"
	#include "Source.h"
	#include "statistics_flt.h"
	#include "statistics_dbl.h"
	
	
	
	
#include <stdbool.h>

#include "common.h"

#include "Stack.h"
#include "Array_dbl.h"
#include "Array_siz.h"
#include "Map.h"
#include "Catalog.h"
#include "LinkerPar.h"
#include "Header.h"
#include "WCS.h"







#define DESTROY  false
#define PRESERVE true







	
	// ----------------------------------------------------------------- //
	// Declaration of private properties and methods of class DataCube   //
	// ----------------------------------------------------------------- //
	
	CLASS DataCube
	{
		char   *data;
		size_t  data_size;
		Header *header;
		int     data_type;
		int     word_size;
		size_t  dimension;
		size_t  axis_size[4];
		double  bscale;
		double  bzero;
		bool    verbosity;
	};
	
	
	


	//#include "OpenCL-Headers-master/CL/cl_d3d10.h"
	//#include "OpenCL-Headers-master/CL/cl_d3d11.h"
	//#include "OpenCL-Headers-master/CL/cl_dx9_media_sharing.h"
	//#include "OpenCL-Headers-master/CL/cl_df9_media_sharing_intel.h"
	//#include "OpenCL-Headers-master/CL/cl_egl.h"
	//#include "OpenCL-Headers-master/CL/cl_ext.h"
	//#include "OpenCL-Headers-master/CL/cl_ext_intel.h"
	//#include "OpenCL-Headers-master/CL/cl_gl.h"
	//#include "OpenCL-Headers-master/CL/cl_gl_ext.h"
	//#include "OpenCL-Headers-master/CL/cl_half.h"
	//#include "OpenCL-Headers-master/CL/cl_icd.h"
	//#include "OpenCL-Headers-master/CL/cl_layer.h"
	//#include "OpenCL-Headers-master/CL/cl_platform.h"
	//#include "OpenCL-Headers-master/CL/cl_va_api_media_sharing_intel.h"
	
	
	// ----------------------------------------------------------------- //
	// Declaration of private properties and methods of class DataCube   //
	// ----------------------------------------------------------------- //

	







//PUBLIC void DataCube_run_scfind_CL(const DataCube *self, DataCube *maskCube, const Array_dbl *kernels_spat, const Array_siz *kernels_spec, const double threshold, const double maskScaleXY, const noise_stat method, const int range, const int scaleNoise, const size_t snWindowXY, const size_t snWindowZ, const size_t snGridXY, const size_t snGridZ, const bool snInterpol, const time_t start_time, const clock_t start_clock);


PUBLIC void DataCube_run_scfind_CL(const DataCube *self, DataCube *maskCube, const Array_dbl *kernels_spat, const Array_siz *kernels_spec, const double threshold, const double maskScaleXY, const noise_stat method, const int range, const int scaleNoise, const size_t snWindowXY, const size_t snWindowZ, const size_t snGridXY, const size_t snGridZ, const bool snInterpol, const time_t start_time, const clock_t start_clock);

//PUBLIC void DataCube_gaussian_filter_CL(DataCube *cube, const double sigma, size_t sizeX, size_t sizeY, size_t sizeZ);
PUBLIC void DataCube_gaussian_filter_CL(DataCube *cube, const double sigma, size_t sizeX, size_t sizeY, size_t sizeZ, size_t cubeSize, cl_context contexto, cl_command_queue colaComando, cl_mem cubeBuffer, cl_kernel kernel1, cl_kernel kernel2);

PUBLIC void DataCube_boxcar_filter_CL(DataCube *cube, size_t radius, size_t sizeX, size_t sizeY, size_t sizeZ, size_t cubeSize, cl_context contexto, cl_command_queue colaComando, cl_mem cubeBuffer, cl_kernel kernel);

//void filter_boxcar_1d_flt_CL(float *data, float *dataCpyArray, const size_t size, const size_t filter_radius, size_t stride);

// Spatial and spectral smoothing
//PUBLIC void DataCube_boxcar_filter_CL(char *self, size_t radius);
//PUBLIC void DataCube_gaussian_filter_CL(char *self, const double sigma);



#endif




