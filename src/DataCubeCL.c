#include "DataCubeKernel.h"



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ----------------------------------------------------------------- //
// Run Smooth + Clip (S+C) finder on data cube                       //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self         - Data cube to run the S+C finder on.          //
//   (2) maskCube     - Mask cube for recording detected pixels.     //
//   (3) kernels_spat - List of spatial smoothing lengths correspon- //
//                      ding to the FWHM of the Gaussian kernels to  //
//                      be applied; 0 = no smoothing.                //
//   (4) kernels_spec - List of spectral smoothing lengths corre-    //
//                      sponding to the widths of the boxcar filters //
//                      to be applied. Must be odd or 0.             //
//   (5) threshold    - Relative flux threshold to be applied.       //
//   (6) maskScaleXY  - Already detected pixels will be set to this  //
//                      value times the original rms of the data be- //
//                      fore smoothing the data again.               //
//   (7) method        - Method to use for measuring the noise in    //
//                      the smoothed copies of the cube; can be      //
//                      NOISE_STAT_STD, NOISE_STAT_MAD or            //
//                      NOISE_STAT_GAUSS for standard deviation,     //
//                      median absolute deviation and Gaussian fit   //
//                      to flux histogram, respectively.             //
//   (8) range        - Flux range to used in noise measurement, Can //
//                      be -1, 0 or 1 for negative only, all or po-  //
//                      sitive only.                                 //
//   (9) scaleNoise   - 0 = no noise scaling; 1 = global noise sca-  //
//                      ling; 2 = local noise scaling. Applied after //
//                      each smoothing operation.                    //
//  (10) snWindowXY   - Spatial window size for local noise scaling. //
//                      See DataCube_scale_noise_local() for de-     //
//                      tails.                                       //
//  (11) snWindowZ    - Spectral window size for local noise sca-    //
//                      ling. See DataCube_scale_noise_local() for   //
//                      details.                                     //
//  (12) snGridXY     - Spatial grid size for local noise scaling.   //
//                      See DataCube_scale_noise_local() for de-     //
//                      tails.                                       //
//  (13) snGridZ      - Spectral grid size for local noise scaling.  //
//                      See DataCube_scale_noise_local() for de-     //
//                      tails.                                       //
//  (14) snInterpol   - Enable interpolation for local noise scaling //
//                      if true. See DataCube_scale_noise_local()    //
//                      for details.                                 //
//  (15) start_time   - Arbitrary time stamp; progress time of the   //
//                      algorithm will be calculated and printed re- //
//                      lative to start_time.                        //
//  (16) start_clock  - Arbitrary clock count; progress time of the  //
//                      algorithm in term of CPU time will be calcu- //
//                      lated and printed relative to clock_time.    //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for running the Smooth + Clip (S+C) finder on the //
//   specified data cube. The S+C finder will smooth the data on the //
//   specified spatial and spectral scales, applying a Gaussian fil- //
//   ter in the spatial domain and a boxcar filter in the spectral   //
//   domain. It will then measure the noise level in each iteration  //
//   and mark all pixels with absolute values greater than or equal  //
//   to the specified threshold (relative to the noise level) as 1   //
//   in the specified mask cube, which must be of 8-bit integer      //
//   type, while non-detected pixels will be set to a value of 0.    //
//   Pixels already detected in a previous iteration will be set to  //
//   maskScaleXY times the original rms noise level of the data be-  //
//   fore smoothing.                                                 //
//   The input data cube must be a 32 or 64 bit floating point data  //
//   array. The spatial kernel sizes must be positive floating point //
//   values that represent the FWHM of the Gaussian kernels to be    //
//   applied in the spatial domain. The spectral kernel sizes must   //
//   be positive, odd integer numbers representing the widths of the //
//   boxcar filters to be applied in the spectral domain. The thre-  //
//   shold is relative to the noise level and should be a floating   //
//   point number greater than about 3.0. Lastly, the value of       //
//   maskScaleXY times the original rms of the data will be used to  //
//   replace pixels in the data cube that were already detected in a //
//   previous iteration. This is to ensure that any sources in the   //
//   data will not be smeared out beyond the extent of the source    //
//   when convolving with large kernel sizes.                        //
//   Several methods are available for measuring the noise in the    //
//   data cube, including the standard deviation, median absolute    //
//   deviation and a Gaussian fit to the flux histogram. These dif-  //
//   fer in their speed and robustness. In addition, the flux range  //
//   used in the noise measurement can be restricted to negative or  //
//   positive pixels only to reduce the impact or actual emission or //
//   absorption featured on the noise measurement.                   //
// ----------------------------------------------------------------- //
PUBLIC void DataCube_run_scfind_CL(const DataCube *self, DataCube *maskCube, const Array_dbl *kernels_spat, const Array_siz *kernels_spec, const double threshold, const double maskScaleXY, const noise_stat method, const int range, const int scaleNoise, const size_t snWindowXY, const size_t snWindowZ, const size_t snGridXY, const size_t snGridZ, const bool snInterpol, const time_t start_time, const clock_t start_clock){


//--------------------------//
	//CONFIGURACIÓN DE OPENCL//
	//-------------------------//
	//Declaración de variables de configuración
	cl_int errNum;//Para depurar
	cl_platform_id *platformIDs;
	cl_platform_id platform;
	cl_device_id  *deviceIDs;
	cl_device_id device;
	cl_uint numPlatforms;
	cl_uint numDevices;
	cl_context context= NULL;
	cl_program program;
	cl_command_queue commandQueue;
	
	
	///Obtener plataforma disponible
	errNum=clGetPlatformIDs(0, NULL, &numPlatforms);//numero de plataformas disponibles
	if(numPlatforms<1 || errNum != CL_SUCCESS)
	{
	printf("\nNo se ha encontrado plataforma\n");
	fflush(stdout);
	return;
	}
	
	platformIDs= (cl_platform_id *) calloc(numPlatforms, sizeof(cl_platform_id));//allocate memory for the installed platforms
	
	
	errNum=clGetPlatformIDs(numPlatforms, platformIDs,NULL);//obtengo lista de las plataformas disponibles
	if (errNum != CL_SUCCESS){
	printf("\nFailed to find any OpenCL platforms\n");
	fflush(stdout);
	return;
	}

	platform=platformIDs[0];
	
	//Obtener device GPU disponible
	errNum=clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);//encuentra número de devices GPU
	if(numDevices<1 || errNum != CL_SUCCESS)
	{
	printf("\nNo se ha encontrado GPU en la plataforma\n");
	fflush(stdout);
	exit(1);
	}
	
	deviceIDs=(cl_device_id *) calloc(numDevices,sizeof(cl_device_id));
	
	
	errNum= clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU,numDevices,&deviceIDs[0],NULL);//solo listar el primer GPU device encontrado en la plataforma
	device=deviceIDs[0];
	
	//Creacion del context para el manejo de colas, memoria y kernels
	//creacion de las propiedades del contexto que se le pasaran como argumento a la función que lo crea
	cl_context_properties contextProperties []={CL_CONTEXT_PLATFORM, (cl_context_properties)platform,0}; 
	//creamos context con estas propiedades
	context= clCreateContext(contextProperties,numDevices,deviceIDs,NULL,NULL,&errNum);
	if (errNum != CL_SUCCESS){
	printf("\nFailed creating context\n");
	fflush(stdout);
	return;
	}
	
	//Creación del commandQueue
	commandQueue=clCreateCommandQueueWithProperties(context,device,0,&errNum);
	//commandQueue=clCreateCommandQueue(context,device,0,&errNum);
	if (errNum != CL_SUCCESS){
	printf("\nFailed creating commandQueue\n");
	fflush(stdout);
	return;
	}
	

	
	//Creación del programa

	const char* cFilename = "src/DataCubeKernel.cl";
	//const char* cFilename = "/home/cris/Escritorio/TFG/SoFiA-openCL/src/DataCubeKernel.cl";
	const char* cPreamble="";
	size_t* szFinalLength=NULL;
	
		
	 //implementacion de la función char* oclLoadProgSource(const char* cFilename, const char* cPreamble, size_t* szFinalLength) del opencl_utils.cpp
	
	 
 // locals
	FILE* pFileStream = NULL;
	size_t szSourceLength;

// open the OpenCL source code file
// Linux version
	pFileStream = fopen(cFilename, "rb");
	if(pFileStream == 0)
	{
	return ;
	}


	size_t szPreambleLength = strlen(cPreamble);

// get the length of the source code
	fseek(pFileStream, 0, SEEK_END);
	szSourceLength = ftell(pFileStream);
	fseek(pFileStream, 0, SEEK_SET);

// allocate a buffer for the source code string and read it in
	char* cSourceString = (char *)malloc(szSourceLength + szPreambleLength + 1);
	memcpy(cSourceString, cPreamble, szPreambleLength);
	if (fread((cSourceString) + szPreambleLength, szSourceLength, 1, pFileStream) != 1)
	{
	fclose(pFileStream);
	free(cSourceString);
	return ;
	}

// close the file and return the total length of the combined (preamble + source) string
	fclose(pFileStream);
	if(szFinalLength != 0)
	{
	*szFinalLength = szSourceLength + szPreambleLength;
	}
	cSourceString[szSourceLength + szPreambleLength] = '\0';
 
 
 	//en cSourceString esta el char que debemos usar para crear y construir el program
 	
 	// Create the program
	program = clCreateProgramWithSource(context, 1, (const char **)&cSourceString, szFinalLength, &errNum);

	if(program==NULL){
	printf("\nFailed to create program\n");
	fflush(stdout);
	return ;
	}


	//Construcción del programa creado
	errNum=clBuildProgram(program,0,NULL,NULL,NULL,NULL);

	if (errNum != CL_SUCCESS){
	printf("\nFailed building program\n");
	fflush(stdout);
	size_t log_size;
	clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,0,NULL,&log_size);
	char *log= (char *) malloc(log_size);
	clGetProgramBuildInfo(program,device,CL_PROGRAM_BUILD_LOG,log_size,log,NULL);
	printf("%s\n", log);
	return;
	}





	//Creación de Kernels
	const char *kernel_name_1="DataCube_set_masked_8_opencl";
	const char *kernel_name_2="DataCube_copy_blanked_opencl";
	const char *kernel_name_3="DataCube_mask_8_opencl";
	const char *kernel_name_4="Xfilters_flt_opencl";
	const char *kernel_name_5="Yfilters_flt_opencl";
	const char *kernel_name_6="Zfilters_flt_opencl";

	cl_kernel OpenCL_DataCube_set_masked_8=clCreateKernel(program,kernel_name_1,&errNum);
	if (OpenCL_DataCube_set_masked_8 == (cl_kernel)NULL || errNum != CL_SUCCESS){
	printf("\nFailed creating kernel\n");
	fflush(stdout);
	return;
	}
	cl_kernel OpenCL_DataCube_copy_blanked=clCreateKernel(program,kernel_name_2,&errNum);
	if (OpenCL_DataCube_copy_blanked == (cl_kernel)NULL || errNum != CL_SUCCESS){
	printf("\nFailed creating kernel\n");
	fflush(stdout);
	return;
	}

	cl_kernel OpenCL_DataCube_mask_8=clCreateKernel(program,kernel_name_3,&errNum);

	cl_kernel OpenCL_Xfilters_flt= clCreateKernel(program, kernel_name_4, &errNum);
	cl_kernel OpenCL_Yfilters_flt= clCreateKernel(program, kernel_name_5, &errNum);
	cl_kernel OpenCL_Zfilters_flt= clCreateKernel(program, kernel_name_6, &errNum);
	
	
	
//////////////FIN CONFIGURACIÓN OPENCL//////////////////////////////////////////////////////////////////////////////////////////////////
	
	
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	
// Sanity checks
	check_null(self);
	check_null(self->data);
	ensure(self->data_type < 0, "The S+C finder can only be applied to floating-point data.");
	check_null(maskCube);
	check_null(maskCube->data);
	ensure(maskCube->data_type == 8, "Mask cube must be of 8-bit integer type.");
	ensure(self->axis_size[0] == maskCube->axis_size[0] && self->axis_size[1] == maskCube->axis_size[1] && self->axis_size[2] == maskCube->axis_size[2], "Data cube and mask cube have different sizes.");
	check_null(kernels_spat);
	check_null(kernels_spec);
	ensure(Array_dbl_get_size(kernels_spat) && Array_siz_get_size(kernels_spec), "Invalid spatial or spectral kernel list encountered.");
	ensure(threshold >= 0.0, "Negative flux threshold encountered.");
	ensure(method == NOISE_STAT_STD || method == NOISE_STAT_MAD || method == NOISE_STAT_GAUSS, "Invalid noise measurement method: %d.", method);
	
	// A few additional settings
	const double FWHM_CONST = 2.0 * sqrt(2.0 * log(2.0));  // Conversion between sigma and FWHM of Gaussian function
	size_t cadence = self->data_size / NOISE_SAMPLE_SIZE;  // Stride for noise calculation
	if(cadence < 2) cadence = 1;
	else if(cadence % self->axis_size[0] == 0) cadence -= 1;    // Ensure stride is not equal to multiple of x-axis size
	message("Using a stride of %zu in noise measurement.\n", cadence);
	
///AÑADIDO/////////////////////////////////////////////////
// Print time: start of the Filter
	timestamp(start_time, start_clock);	
	
	


	int *size_x =  (int *)&(self->axis_size[0]);
	int *size_y= (int *)&(self->axis_size[1]);
	int *size_z=(int *)&(self->axis_size[2]);
	//array con cantidad de hilos en cada dimensión del DataCube
	size_t globalWorkSize[3]={self->axis_size[0],self->axis_size[1],self->axis_size[2]};
	//array con cantidad de hilos en los que agruparé los hilos de cada dimensión
	//size_t localWorkSize[3]={8,8,4};
	
	

	//array para almacenar los eventos generados por los kernels.
	cl_event K_events[7]; 

	float value;
	float umbral;
	unsigned char maskValue=1;
	
//Creación de los buffer
//Tamaños de cubos y máscara
	size_t size = (size_t) (self->data_size * self->word_size * sizeof(char));
	size_t sizeMask = (size_t) (maskCube->data_size * maskCube->word_size * sizeof(char));
//Búferes del cubo de datos y de la máscara
				
	cl_mem mascara= clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR , sizeMask, maskCube->data, &errNum);
	cl_mem cuboFuente= clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR ,size,self->data, &errNum );		
	//cl_mem mascara= clCreateBuffer(context, CL_MEM_READ_WRITE , sizeMask, NULL, &errNum);
	//cl_mem cuboFuente= clCreateBuffer(context, CL_MEM_READ_WRITE ,size,NULL, &errNum );
	cl_mem  cuboSalida= clCreateBuffer(context, CL_MEM_READ_WRITE , size, NULL , &errNum );
	

			
////////////////////////////////////////////////////////////////////	
	
	
	// Measure noise in original cube with sampling "cadence"
	double rms;
	double rms_smooth;
	
	if(method == NOISE_STAT_STD)      rms = DataCube_stat_std(self, 0.0, cadence, range);
	else if(method == NOISE_STAT_MAD) rms = MAD_TO_STD * DataCube_stat_mad(self, 0.0, cadence, range);
	else                              rms = DataCube_stat_gauss(self, cadence, range);
	
	// Run S+C finder for all smoothing kernels
	//size_t i;
	for(size_t i = 0; i < Array_dbl_get_size(kernels_spat); ++i)
	{
		//size_t j;
		for(size_t j = 0; j < Array_siz_get_size(kernels_spec); ++j)
		{
			message("Smoothing kernel:  [%.1f] x [%zu]", Array_dbl_get(kernels_spat, i), Array_siz_get(kernels_spec, j));
			
			// Check if any smoothing requested
			if(Array_dbl_get(kernels_spat, i) || Array_siz_get(kernels_spec, j))
			{
				// Smoothing required; create a copy of the original cube
				DataCube *smoothedCube = DataCube_copy(self);
				
 
				// Set flux of already detected pixels to maskScaleXY * rms
				//AÑADIDO///////////////////////////
				
				//cl_int par_value;//PARA CONSULTAR EL STATUS DE LOS EVENTOS
							
			
				value= (float)(maskScaleXY * rms);
				
		
							
				
					
					
// --------------------------------------------------------
    // Start Core sequence... copy input data to GPU, compute, copy results back					
					//Escritura de buffer de entrada al kernel	
					 // Asynchronous write of data to GPU device
					 
	
	
				//if(maskScaleXY >= 0.0){
					//DataCube_set_masked_8(smoothedCube, maskCube, maskScaleXY * rms);


					//ESCRITURA DE BUFFERS OpenCL_DataCube_set_masked_8
					/*errNum= clEnqueueWriteBuffer(commandQueue,mascara,CL_TRUE,0,sizeMask,maskCube->data,0,NULL,&K_events[0]);
					errNum = clWaitForEvents(1, &K_events[0]);
					
					errNum|= clEnqueueWriteBuffer(commandQueue,cuboFuente,CL_TRUE,0,size,self->data,0,NULL,&K_events[1]);		
					errNum = clWaitForEvents(1, &K_events[1]);
*/

					//Poner argumentos al kernel  OpenCL_DataCube_set_masked_8
					errNum=clSetKernelArg(OpenCL_DataCube_set_masked_8,0, sizeof(cl_mem), (void*)&cuboSalida);	
					errNum |=clSetKernelArg(OpenCL_DataCube_set_masked_8,1, sizeof(cl_mem), (void*)&mascara);
					errNum |=clSetKernelArg(OpenCL_DataCube_set_masked_8,2, sizeof(cl_mem), (void*)&cuboFuente);
					errNum |=clSetKernelArg(OpenCL_DataCube_set_masked_8,3, sizeof(float), &value);
					errNum |=clSetKernelArg(OpenCL_DataCube_set_masked_8,4, sizeof(int), &size_x);
					errNum |=clSetKernelArg(OpenCL_DataCube_set_masked_8,5, sizeof(int), &size_y);
					errNum |=clSetKernelArg(OpenCL_DataCube_set_masked_8,6, sizeof(int), &size_z);

					if (errNum != CL_SUCCESS){
						printf("\nERROR AL PONER ARGUMENTOS AL KERNEL\n");
						fflush(stdout);
						return ;
					}


					//INVOCAMOS KERNEL OpenCL_DataCube_set_masked_8, APLICA MÁSCARA A LA COPIA DEL CUBO ORIGINAL

					errNum=clEnqueueNDRangeKernel(commandQueue, OpenCL_DataCube_set_masked_8, 3, NULL, globalWorkSize, NULL, 0, NULL, &K_events[0]);
					//errNum=clEnqueueNDRangeKernel(commandQueue, OpenCL_DataCube_set_masked_8, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, &K_events[2]);

							
					if (errNum != CL_SUCCESS){
					printf("\nError de lanzamiento del kernel OpenCL_DataCube_set_masked_8.\n");
					fflush(stdout);									
					if(errNum ==CL_INVALID_WORK_GROUP_SIZE){
					printf("\nCL_INVALID_WORK_GROUP_SIZE.\n");
					fflush(stdout);
					return;}
					else if (errNum==CL_INVALID_WORK_ITEM_SIZE){
					printf("\nCL_INVALID_WORK_ITEM_SIZE.\n");
					fflush(stdout);
					return;}
					else if (errNum==CL_OUT_OF_RESOURCES){
					printf("\nCL_OUT_OF_RESOURCES.\n");
					fflush(stdout);
					return;}
					else if (errNum==CL_MEM_OBJECT_ALLOCATION_FAILURE){
					printf("\nCL_MEM_OBJECT_ALLOCATION_FAILURE.\n");
					fflush(stdout);
					return;}
					else if (errNum==CL_INVALID_EVENT_WAIT_LIST){
					printf("\nCL_INVALID_EVENT_WAIT_LIST.\n");
					fflush(stdout);
					return;}
					else if (errNum==CL_OUT_OF_HOST_MEMORY ){
					printf("\nCL_OUT_OF_HOST_MEMORY.\n");
					fflush(stdout);
					return;}
					/*else if (errNum==){}
					else if (errNum==){}
					else if (errNum==){}
					else if (errNum==){}
					else if (errNum==){}*/
					else{
					printf("\nnope.\n");
					fflush(stdout);
					return;
					}
					}


					errNum = clWaitForEvents(1, &K_events[0]);
					
					//LECTURA DEL BUFFER CON DATOS DEL CUBO PASADOS POR LA MÁSCARA
				//	errNum=clEnqueueReadBuffer(commandQueue, cuboSalida, CL_TRUE, 0, size, smoothedCube->data, 0, NULL, &K_events[3]);

					clFinish(commandQueue);

				/*	errNum = clWaitForEvents(1, &K_events[3]);
					if (errNum != CL_SUCCESS)
					{
					printf("clWaitForEvents tras lectura del buffer (returned %d).\n", errNum);
					fflush(stdout);
					}

	*/


				//}
	
	
				
				// Spatial and spectral smoothing
				if(Array_dbl_get(kernels_spat, i) > 0.0){
				 DataCube_gaussian_filter_CL(smoothedCube, Array_dbl_get(kernels_spat, i) / FWHM_CONST, self->axis_size[0], self->axis_size[1], self->axis_size[2], size, context, commandQueue, cuboSalida, OpenCL_Xfilters_flt, OpenCL_Yfilters_flt);
				 }
				 
				if(Array_siz_get(kernels_spec, j) > 0) {
				 DataCube_boxcar_filter_CL(smoothedCube, Array_siz_get(kernels_spec, j) / 2,  self->axis_size[0], self->axis_size[1], self->axis_size[2], size, context , commandQueue, cuboSalida, OpenCL_Zfilters_flt);
				 }
				/*if(Array_dbl_get(kernels_spat, i) > 0.0) DataCube_gaussian_filter(smoothedCube, Array_dbl_get(kernels_spat, i) / FWHM_CONST);*/
				/*if(Array_siz_get(kernels_spec, j) > 0)   DataCube_boxcar_filter(smoothedCube, Array_siz_get(kernels_spec, j) / 2);*/





				// Copy original blanks into smoothed cube again
				// (these were set to 0 during smoothing)
				
				//Sobreescribo el buffer cuboSalida con los nuevos datos del smoothedCube sacado de los filtros
				/*errNum|= clEnqueueWriteBuffer(commandQueue,cuboSalida,CL_TRUE,0,size,smoothedCube->data,0,NULL,&K_events[4]);		
				errNum = clWaitForEvents(1, &K_events[4]);
				*/
				//Argumentos del kernel OpenCL_DataCube_copy_blanked
				errNum=clSetKernelArg(OpenCL_DataCube_copy_blanked,0, sizeof(cl_mem), (void*)&cuboSalida);	
				errNum |=clSetKernelArg(OpenCL_DataCube_copy_blanked,1, sizeof(cl_mem), (void*)&cuboFuente);
				
				errNum=clEnqueueNDRangeKernel(commandQueue, OpenCL_DataCube_copy_blanked, 3, NULL, globalWorkSize, NULL, 0, NULL, &K_events[1]);
				
				//errNum=clEnqueueNDRangeKernel(commandQueue, OpenCL_DataCube_copy_blanked, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, &K_events[1]);
				
				errNum = clWaitForEvents(1, &K_events[1]);
				
				errNum=clEnqueueReadBuffer(commandQueue, cuboSalida, CL_TRUE, 0, size, smoothedCube->data, 0, NULL, &K_events[2]);
				
				clFinish(commandQueue);
				errNum = clWaitForEvents(1, &K_events[2]);
				
				
			//	DataCube_copy_blanked(smoothedCube, self);
				
				// Scale noise if requested
				if(scaleNoise == 1)
				{
							
					message("Correcting for noise variations along spectral axis.\n");
					DataCube_scale_noise_spec(smoothedCube);
				}
				else if(scaleNoise == 2)
				{
				
					message("Correcting for local noise variations.");
					DataCube *noiseCube = DataCube_scale_noise_local(
						smoothedCube,
						snWindowXY,
						snWindowZ,
						snGridXY,
						snGridZ,
						snInterpol
					);
					DataCube_delete(noiseCube);
				}
				
				// Calculate the RMS of the smoothed cube
				
				if(method == NOISE_STAT_STD)      rms_smooth = DataCube_stat_std(smoothedCube, 0.0, cadence, range);
				
				else if(method == NOISE_STAT_MAD) rms_smooth = MAD_TO_STD * DataCube_stat_mad(smoothedCube, 0.0, cadence, range);
				
				else                              rms_smooth = DataCube_stat_gauss(smoothedCube, cadence, range);
				
				message("Noise level:       %.3e", rms_smooth);
				
				// Add pixels above threshold to mask
				
				//Reescribo el buffer cuboSalida con los posibles nuevos datos del smoothedCube
				
				
				/*errNum|= clEnqueueWriteBuffer(commandQueue,cuboSalida,CL_TRUE,0,size,smoothedCube->data,0,NULL,&K_events[7]);		
				errNum = clWaitForEvents(1, &K_events[7]);
				*/
				
				umbral=(float)(threshold * rms_smooth);
				
				
				
				//Argumentos del kernel OpenCL_DataCube_mask_8
				errNum=clSetKernelArg(OpenCL_DataCube_mask_8,0, sizeof(cl_mem), (void*)&cuboSalida);	
				errNum |=clSetKernelArg(OpenCL_DataCube_mask_8,1, sizeof(cl_mem), (void*)&mascara);
				errNum |=clSetKernelArg(OpenCL_DataCube_mask_8,2, sizeof(float), &umbral);
				errNum |=clSetKernelArg(OpenCL_DataCube_mask_8,3, sizeof(unsigned char), &maskValue);
				
				errNum=clEnqueueNDRangeKernel(commandQueue, OpenCL_DataCube_mask_8, 3, NULL, globalWorkSize, NULL, 0, NULL, &K_events[3]);
				//errNum=clEnqueueNDRangeKernel(commandQueue, OpenCL_DataCube_mask_8, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, &K_events[3]);
				
				errNum = clWaitForEvents(1, &K_events[3]);
				
				errNum=clEnqueueReadBuffer(commandQueue, mascara, CL_TRUE, 0, sizeMask, maskCube->data, 0, NULL, &K_events[4]);
				
				clFinish(commandQueue);
				errNum = clWaitForEvents(1, &K_events[4]);
				
				
				
				
				
			//	DataCube_mask_8(smoothedCube, maskCube, threshold * rms_smooth, 1);
				
				// Delete smoothed cube again
				DataCube_delete(smoothedCube);
			}
			else
			{
				// No smoothing required; apply threshold to original cube
				message("Noise level:       %.3e", rms);
				
				
				
				//ESCRITURA DE BUFFERS OpenCL_DataCube_mask_8
				/*errNum= clEnqueueWriteBuffer(commandQueue,mascara,CL_TRUE,0,sizeMask,maskCube->data,0,NULL,&K_events[10]);
				errNum = clWaitForEvents(1, &K_events[10]);
								
				errNum|= clEnqueueWriteBuffer(commandQueue,cuboFuente,CL_TRUE,0,size,self->data,0,NULL,&K_events[11]);		
				errNum = clWaitForEvents(1, &K_events[11]);
				*/			
				
				umbral=(float)(threshold * rms);
				
				//Argumentos del kernel OpenCL_DataCube_mask_8
				errNum=clSetKernelArg(OpenCL_DataCube_mask_8,0, sizeof(cl_mem), (void*)&cuboFuente);	
				errNum |=clSetKernelArg(OpenCL_DataCube_mask_8,1, sizeof(cl_mem), (void*)&mascara);
				errNum |=clSetKernelArg(OpenCL_DataCube_mask_8,2, sizeof(float), &umbral);
				errNum |=clSetKernelArg(OpenCL_DataCube_mask_8,3, sizeof(unsigned char), &maskValue);
				
				
				errNum=clEnqueueNDRangeKernel(commandQueue, OpenCL_DataCube_mask_8, 3, NULL, globalWorkSize, NULL, 0, NULL, &K_events[5]);
				//errNum=clEnqueueNDRangeKernel(commandQueue, OpenCL_DataCube_mask_8, 3, NULL, globalWorkSize, localWorkSize, 0, NULL, &K_events[5]);
				
				errNum = clWaitForEvents(1, &K_events[5]);
				
				errNum=clEnqueueReadBuffer(commandQueue, mascara, CL_TRUE, 0, sizeMask, maskCube->data, 0, NULL, &K_events[6]);
				
				clFinish(commandQueue);
				errNum = clWaitForEvents(1, &K_events[6]);
				
				
				//DataCube_mask_8(self, maskCube, threshold * rms, 1);
			}
			
			// Print time
			timestamp(start_time, start_clock);
		}
	}

	// Cleanup
	
        clReleaseKernel( OpenCL_DataCube_set_masked_8);
        clReleaseKernel( OpenCL_DataCube_copy_blanked);
    	clReleaseKernel( OpenCL_DataCube_mask_8);
    	clReleaseKernel( OpenCL_Xfilters_flt);
    	clReleaseKernel( OpenCL_Yfilters_flt);
    	clReleaseKernel( OpenCL_Zfilters_flt);
    
        clReleaseProgram(program);

        free(cSourceString);
   
        clReleaseMemObject(mascara);
   
        clReleaseMemObject(cuboFuente);
  
        clReleaseMemObject(cuboSalida);
        
	
	

        clReleaseCommandQueue(commandQueue);
   
        free(deviceIDs);
   
        clReleaseContext(context);
	
	return;
	}
	
	
	
	
// ----------------------------------------------------------------- //
// Apply 2D Gaussian filter to spatial planes                        //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self    - Object self-reference.                            //
//   (2) sigma   - Standard deviation of the Gaussian in pixels.     //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for convolving each spatial image plane (x-y) of  //
//   the data cube with a Gaussian function of standard deviation    //
//   sigma. The Gaussian convolution is approximated through a set   //
//   of 1D boxcar filters, which makes the algorithm extremely fast. //
//   Limitations from this approach are that the resulting convolu-  //
//   tion kernel is only an approximation of a Gaussian (although a  //
//   fairly accurate one) and the value of sigma can only be appro-  //
//   ximated (typically within +/- 0.2 sigma) and must be at least   //
//   1.5 pixels.                                                     //
//   The algorithm is NaN-safe by setting all NaN values to 0. Any   //
//   pixel outside of the image boundaries is also assumed to be 0.  //
// ----------------------------------------------------------------- //

PUBLIC void DataCube_gaussian_filter_CL(DataCube *cube, const double sigma, size_t sizeX, size_t sizeY, size_t sizeZ, size_t cubeSize, cl_context contexto, cl_command_queue colaComando, cl_mem cubeBuffer, cl_kernel kernel1, cl_kernel kernel2)
{
	
	// Set up parameters required for boxcar filter
	size_t n_iter;
	size_t filter_radius;
	optimal_filter_size_dbl(sigma, &filter_radius, &n_iter);
	int size_x =  (int)sizeX;
	int size_y =  (int)sizeY;
	int radius_filter=(int) filter_radius;
	cl_int numErr;
	//cl_event gaussian_events[2]; //eventos de esta función
	cl_event X_filter_events[n_iter + 1]; //sustituir por n_iter
	cl_event Y_filter_events[n_iter + 1]; //sustituir por n_iter
	
	//hilos en cada dimensión a recorrer por las funciones de filtrado 
	size_t xFilterGlobalWorkSize[3]={1,sizeY,sizeZ};
	//size_t xFilterLocalWorkSize[3]={1,16,16};
	
	size_t yFilterGlobalWorkSize[3]={sizeX,1,sizeZ};
	//size_t yFilterLocalWorkSize[3]={16,1,16};

	//Creo el buffer para dataRadiusX
	float *dataRadiusX;
	dataRadiusX= (float *) calloc(sizeY * sizeZ * (filter_radius + 2), sizeof(float));
	size_t xRadiusSize = (size_t) (sizeY * sizeZ * (filter_radius + 2) * sizeof(float));
	
	//cl_mem  XRadiusData= clCreateBuffer(contexto, CL_MEM_READ_WRITE, xRadiusSize, NULL , &numErr );
	cl_mem  XRadiusData= clCreateBuffer(contexto, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, xRadiusSize, dataRadiusX , &numErr );
	
	
	//Establezco argumentos del kernel Xfilters
	//errNum=clSetKernelArg(kernel1,0, sizeof(cl_mem), (void*)&cuboSalida);
	numErr=clSetKernelArg(kernel1,0, sizeof(cl_mem), (void*)&cubeBuffer);		
	numErr |=clSetKernelArg(kernel1,1, sizeof(cl_mem), (void*)&XRadiusData);
	numErr |=clSetKernelArg(kernel1,2, sizeof(int), &radius_filter);
	numErr |=clSetKernelArg(kernel1,3, sizeof(int), &size_x);

	//size_t i;
	for(size_t i = n_iter; i--;){
		//Lanzo el kernel, se lanza n_iter veces
		//kernel1= OpenCL_Xfilters_flt
		numErr=clEnqueueNDRangeKernel(colaComando, kernel1, 3, NULL, xFilterGlobalWorkSize, NULL, 0, NULL, &X_filter_events[i]);
		
	/*	if(numErr !=CL_SUCCESS){
		printf("ERROR AL LANZAR EL KERNEL\n");
	fflush(stdout);
		if(numErr ==CL_INVALID_PROGRAM_EXECUTABLE){
		printf("CL_INVALID_PROGRAM_EXECUTABLE\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_COMMAND_QUEUE){printf("CL_INVALID_COMMAND_QUEUE\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_KERNEL){printf("CL_INVALID_KERNEL\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_CONTEXT){printf("CL_INVALID_CONTEXT\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_KERNEL_ARGS){printf("CL_INVALID_KERNEL_ARGS\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_WORK_DIMENSION){printf("CL_INVALID_WORK_DIMENSION\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_WORK_GROUP_SIZE){printf("CL_INVALID_WORK_GROUP_SIZE\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_WORK_GROUP_SIZE){printf("CL_INVALID_WORK_GROUP_SIZE\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_WORK_GROUP_SIZE){printf("CL_INVALID_WORK_GROUP_SIZE\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_WORK_ITEM_SIZE){printf("CL_INVALID_WORK_ITEM_SIZE\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_GLOBAL_OFFSET){printf("CL_INVALID_GLOBAL_OFFSET\n");
	fflush(stdout);}
		else if(numErr ==CL_OUT_OF_RESOURCES){printf("CL_OUT_OF_RESOURCES\n");
	fflush(stdout);}
		else if(numErr ==CL_MEM_OBJECT_ALLOCATION_FAILURE){printf("CL_MEM_OBJECT_ALLOCATION_FAILURE\n");
	fflush(stdout);}
		else if(numErr ==CL_INVALID_EVENT_WAIT_LIST){printf("CL_INVALID_EVENT_WAIT_LIST\n");
	fflush(stdout);}
		else if(numErr ==CL_OUT_OF_HOST_MEMORY){printf("CL_OUT_OF_HOST_MEMORY\n");
	fflush(stdout);}
		else{printf("nope\n");
	fflush(stdout);}
																												
	}*/
				
		numErr = clWaitForEvents(1, &X_filter_events[i]);
		
	}
	
	free(dataRadiusX);
	
	float *dataRadiusY;
	dataRadiusY= (float *) calloc(sizeX * sizeZ * (filter_radius + 2), sizeof(float));
	size_t yRadiusSize = (size_t) (sizeX * sizeZ * (filter_radius + 2) * sizeof(float));
	
	//cl_mem  YRadiusData= clCreateBuffer(contexto, CL_MEM_READ_WRITE , yRadiusSize, NULL , &numErr );
	cl_mem  YRadiusData= clCreateBuffer(contexto, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, yRadiusSize, dataRadiusY , &numErr );
	//Escribo en buffer lo que sea apuntado por dataRadiusX
	/*numErr= clEnqueueWriteBuffer(colaComando,YRadiusData,CL_TRUE,0,yRadiusSize,dataRadiusY,0,NULL,&gaussian_events[1]);
	numErr = clWaitForEvents(1, &gaussian_events[1]);
	*/
	
	//Establezco argumentos del kernel Yfilters
	//kernel2= OpenCL_Yfilters_flt
	numErr=clSetKernelArg(kernel2,0, sizeof(cl_mem), (void*)&cubeBuffer);	
	numErr |=clSetKernelArg(kernel2,1, sizeof(cl_mem), (void*)&YRadiusData);
	numErr |=clSetKernelArg(kernel2,2, sizeof(int), &radius_filter);
	numErr |=clSetKernelArg(kernel2,3, sizeof(int), &size_y);

	
	//size_t i;
	for(size_t i = n_iter; i--;){
	//Lanzo el kernel, se lanza n_iter veces
		numErr=clEnqueueNDRangeKernel(colaComando, kernel2, 3, NULL, yFilterGlobalWorkSize, NULL, 0, NULL, &Y_filter_events[i]);
				
		numErr = clWaitForEvents(1, &Y_filter_events[i]);
	
	}
	
	free(dataRadiusY);
	
	
	numErr=clEnqueueReadBuffer(colaComando, cubeBuffer, CL_TRUE, 0, cubeSize, cube->data, 0, NULL, NULL);
				
	clFinish(colaComando);
				
	clReleaseMemObject(XRadiusData);

	clReleaseMemObject(YRadiusData);
		

	return;
}

	
// ----------------------------------------------------------------- //
// Apply boxcar filter to spectral axis                              //
// ----------------------------------------------------------------- //
// Arguments:                                                        //
//                                                                   //
//   (1) self    - Object self-reference.                            //
//   (2) radius  - Filter radius in channels.                        //
//                                                                   //
// Return value:                                                     //
//                                                                   //
//   No return value.                                                //
//                                                                   //
// Description:                                                      //
//                                                                   //
//   Public method for convolving each spectrum of the data cube     //
//   with a boxcar filter of size 2 * radius + 1. The algorithm is   //
//   NaN-safe by setting all NaN values to 0 prior to filtering. Any //
//   pixel outside of the cube's spectral range is also assumed to   //
//   be 0.                                                           //
// ----------------------------------------------------------------- //	
PUBLIC void DataCube_boxcar_filter_CL(DataCube *cube, size_t radius, size_t sizeX, size_t sizeY, size_t sizeZ, size_t cubeSize, cl_context contexto, cl_command_queue colaComando, cl_mem cubeBuffer, cl_kernel kernel)
{
	// Sanity checks
	if(radius < 1) radius = 1;

	int size_z =  (int)sizeZ;
	int radiusData= (int)radius;
	
	cl_int numErr;
	cl_event boxcar_events[2]; //eventos de esta función
	
	
	//hilos en cada dimensión a recorrer por las funciones de filtrado 
	size_t zFilterGlobalWorkSize[3]={sizeX,sizeY,1};
	
	
	//Creo el buffer para dataRadiusZ
	float *dataRadiusZ;
	dataRadiusZ= (float *) calloc(sizeY * sizeX * (radius + 2), sizeof(float));
	size_t zRadiusSize = (size_t) (sizeY * sizeX * (radius + 2) * sizeof(float));
	
	//cl_mem  ZRadiusData= clCreateBuffer(contexto, CL_MEM_READ_WRITE , zRadiusSize, NULL , &numErr );
	cl_mem  ZRadiusData= clCreateBuffer(contexto, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, zRadiusSize, dataRadiusZ , &numErr );
	//Escribo en buffer lo que sea apuntado por dataRadiusX
	/*numErr= clEnqueueWriteBuffer(colaComando,ZRadiusData,CL_TRUE,0,zRadiusSize,dataRadiusZ,0,NULL,&boxcar_events[0]);
	numErr = clWaitForEvents(1, &boxcar_events[0]);
	*/
	//Establezco argumentos del kernel Zfilters
	//errNum=clSetKernelArg(kernel,0, sizeof(cl_mem), (void*)&cuboSalida);
	numErr=clSetKernelArg(kernel,0, sizeof(cl_mem), (void*)&cubeBuffer);		
	numErr |=clSetKernelArg(kernel,1, sizeof(cl_mem), (void*)&ZRadiusData);
	numErr |=clSetKernelArg(kernel,2, sizeof(int), &radiusData);
	numErr |=clSetKernelArg(kernel,3, sizeof(int), &size_z);

	numErr=clEnqueueNDRangeKernel(colaComando, kernel, 3, NULL, zFilterGlobalWorkSize, NULL, 0, NULL, &boxcar_events[1]);
	numErr = clWaitForEvents(1, &boxcar_events[1]);
	free(dataRadiusZ);
	
	numErr=clEnqueueReadBuffer(colaComando, cubeBuffer, CL_TRUE, 0, cubeSize, cube->data, 0, NULL, NULL);
	
	clFinish(colaComando);
				
	clReleaseMemObject(ZRadiusData);	
	
	return;
}
	
	
	
	
	
	
	
	
	
	
	
	

	
	
	
	
	

