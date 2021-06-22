
__kernel void DataCube_set_masked_8_opencl(__global float *cube, __global char *cubeMask,__global float *sourceCube, const float value, const int sizeX, const int sizeY,const int sizeZ)
{
	
	size_t id= ((get_global_id(2) * get_global_size(0) * get_global_size(1))+(get_global_id(1) * get_global_size(0)) + get_global_id(0));


	if(cubeMask[id] !=0){
	  	
	  		cube[id]= copysign(value, sourceCube[id]);			
	 }else{
	 	
	  		
  		if(isnan(sourceCube[id])){
  		
  			cube[id]=0.0;
  		}else{
  		
  			cube[id]= sourceCube[id];
  		}	
	 } 			
	 
	return;
}




__kernel void DataCube_copy_blanked_opencl(__global float *cube, __global float *sourceCube)
{
	
	size_t id= ((get_global_id(2) * get_global_size(0) * get_global_size(1))+(get_global_id(1) * get_global_size(0)) + get_global_id(0));



	if(isnan(sourceCube[id])){
		cube[id]= NAN;
	}
	 
	return;
}



__kernel void DataCube_mask_8_opencl(__global float *cubo, __global char *cubeMask, const float threshold, const unsigned char value )
{
	
	size_t id= ((get_global_id(2) * get_global_size(0) * get_global_size(1))+(get_global_id(1) * get_global_size(0)) + get_global_id(0));



	if(fabs(cubo[id])>threshold){
		cubeMask[id]= value;
	}
	 
	return;
}



__kernel void Xfilters_flt_opencl(__global float *cube, __global float *dataRadius, int radius, int sizeX){

	//identificador de la posición inicial al smoothedcube de la fila a analizar:plano depth, fila row.
	size_t pos= (get_global_id(1) * sizeX) + (get_global_id(2) * sizeX * get_global_size(1));

	//identificador para recorrer el dataRadius
	size_t radius_pos=((get_global_id(2) * get_global_size(1)) + get_global_id(1)) * (radius + 2);
	
	//float *data= &(cube[pos]);
	//float *dataCpyArray= &(dataRadius[radius_pos]);
	const size_t size= (size_t)sizeX;
	const size_t filter_radius=(size_t)radius;
	size_t stride=1;
	
	//filter_boxcar_1d_flt_CL(cube[pos] , dataRadius[radius_pos] , (size_t)sizeX , (size_t)radius , 1);
	//void filter_boxcar_1d_flt_CL(float *data, float *dataCpyArray, const size_t size, const size_t filter_radius, size_t stride)
//{	
	// Define filter size
	const size_t filter_size = 2 * filter_radius + 1;
	const float inv_filter_size = 1.0 / filter_size;
	size_t i;

	//float* dataCpyArray = new float[filter_radius + 2];
	for(i = filter_radius + 1; i>0; i--) {
		//dataCpyArray[i] = 0;
		dataRadius[radius_pos + i] = 0;
	}
	int countdataCpyArray = 1;
	//dataCpyArray[0]= data[(size - 1)*stride];
	dataRadius[radius_pos + 0]= cube[pos + ((size - 1)*stride)];
	// Apply boxcar filter to last data point
	for(i = filter_radius; i--;) { //desde el anterior en un radio. El mismo y los posteriores no son necesarios
		int iC = size + i - filter_radius - 1;
		if (iC >= 0){
			//data[(size - 1)*stride] += data[(iC)*stride];
			cube[pos + ((size - 1)*stride)] += cube[pos + ((iC)*stride)];
		}
	}
	//data[(size - 1)*stride] *= inv_filter_size;
	cube[pos + ((size - 1)*stride)] *= inv_filter_size;
	
	// Recursively apply boxcar filter to  all previous data points
	for(i = size - 1; i--;) {
		int iC1 = i - filter_radius;
		int iC2 = i + filter_radius + 1;
		float dataCpy1;
		float dataCpy2;

		//dataCpyArray[countdataCpyArray] = data[i*stride];
		dataRadius[radius_pos + countdataCpyArray] = cube[pos + (i*stride)];
		if (countdataCpyArray >= (int)filter_radius + 1){
			countdataCpyArray = 0;
		}else{
			countdataCpyArray++;
		}

		if (iC1 >= 0) {
			//dataCpy1 = data[iC1*stride];
			dataCpy1 = cube[pos + (iC1*stride)];
		}else{
			dataCpy1 = 0;
		}
		if (iC2 < (int)size) {
			//dataCpy2 = dataCpyArray[countdataCpyArray];
			dataCpy2 = dataRadius[radius_pos + countdataCpyArray];
		}else{
			dataCpy2 = 0;
		}
		//data[i*stride] = data[(i + 1)*stride] + (dataCpy1 - dataCpy2) * inv_filter_size;
		cube[pos+ (i*stride)] = cube[pos + ((i + 1)*stride)] + (dataCpy1 - dataCpy2) * inv_filter_size;
	}
	return;
}





__kernel void Yfilters_flt_opencl(__global float *cube, __global float *dataRadius, int radius, int sizeY){

	//identificador de la posición inicial al smoothedcube de la columna a analizar
	size_t pos= get_global_id(0) + (get_global_id(2) * get_global_size(0) * sizeY);

	//identificador para recorrer el dataRadius
	size_t radius_pos=(get_global_id(0) + (get_global_id(2) * get_global_size(0))) * (radius + 2);
	
	
	const size_t size=get_global_size(0);
	const size_t filter_radius=(size_t)radius;
	size_t stride=get_global_size(0);
	
	//filter_boxcar_1d_flt_CL(cube[pos] , dataRadius[radius_pos] , sizeX , (size_t)radius , sizeX);
	//void filter_boxcar_1d_flt_CL(float *data, float *dataCpyArray, const size_t size, const size_t filter_radius, size_t stride)
//{	
	// Define filter size
	const size_t filter_size = 2 * filter_radius + 1;
	const float inv_filter_size = 1.0 / filter_size;
	size_t i;

	//float* dataCpyArray = new float[filter_radius + 2];
	for(i = filter_radius + 1; i>0; i--) {
		//dataCpyArray[i] = 0;
		dataRadius[radius_pos + i] = 0;
	}
	int countdataCpyArray = 1;
	//dataCpyArray[0]= data[(size - 1)*stride];
	dataRadius[radius_pos + 0]= cube[pos + ((size - 1)*stride)];
	// Apply boxcar filter to last data point
	for(i = filter_radius; i--;) { //desde el anterior en un radio. El mismo y los posteriores no son necesarios
		int iC = size + i - filter_radius - 1;
		if (iC >= 0){
			//data[(size - 1)*stride] += data[(iC)*stride];
			cube[pos + ((size - 1)*stride)] += cube[pos + ((iC)*stride)];
		}
	}
	//data[(size - 1)*stride] *= inv_filter_size;
	cube[pos + ((size - 1)*stride)] *= inv_filter_size;
	
	// Recursively apply boxcar filter to  all previous data points
	for(i = size - 1; i--;) {
		int iC1 = i - filter_radius;
		int iC2 = i + filter_radius + 1;
		float dataCpy1;
		float dataCpy2;

		//dataCpyArray[countdataCpyArray] = data[i*stride];
		dataRadius[radius_pos + countdataCpyArray] = cube[pos + (i*stride)];
		if (countdataCpyArray >= (int)filter_radius + 1){
			countdataCpyArray = 0;
		}else{
			countdataCpyArray++;
		}

		if (iC1 >= 0) {
			//dataCpy1 = data[iC1*stride];
			dataCpy1 = cube[pos + (iC1*stride)];
		}else{
			dataCpy1 = 0;
		}
		if (iC2 < (int)size) {
			//dataCpy2 = dataCpyArray[countdataCpyArray];
			dataCpy2 = dataRadius[radius_pos + countdataCpyArray];
		}else{
			dataCpy2 = 0;
		}
		//data[i*stride] = data[(i + 1)*stride] + (dataCpy1 - dataCpy2) * inv_filter_size;
		cube[pos+ (i*stride)] = cube[pos + ((i + 1)*stride)] + (dataCpy1 - dataCpy2) * inv_filter_size;
	}
	return;
	
	
}


__kernel void Zfilters_flt_opencl(__global float *cube, __global float *dataRadius, int radius, int sizeZ){

	//identificador de la posición inicial al smoothedcube de la columna a analizar
	size_t pos=get_global_id(0) + (get_global_id(1) * get_global_size(0)) ;

	//identificador para recorrer el dataRadius
	size_t radius_pos=(get_global_id(0) + (get_global_id(1) * get_global_size(0))) * (radius + 2);
	
	const size_t size=(size_t)sizeZ;
	const size_t filter_radius=(size_t)radius;
	size_t stride=get_global_size(0) * get_global_size(1);
	
	//filter_boxcar_1d_flt_device(cube[pos], dataRadius[radius_pos], (size_t)sizeZ,(size_t) radius, sizeX * sizeY);
	//void filter_boxcar_1d_flt_CL(float *data, float *dataCpyArray, const size_t size, const size_t filter_radius, size_t stride)
//{	
	// Define filter size
	const size_t filter_size = 2 * filter_radius + 1;
	const float inv_filter_size = 1.0 / filter_size;
	size_t i;

	//float* dataCpyArray = new float[filter_radius + 2];
	for(i = filter_radius + 1; i>0; i--) {
		//dataCpyArray[i] = 0;
		dataRadius[radius_pos + i] = 0;
	}
	int countdataCpyArray = 1;
	//dataCpyArray[0]= data[(size - 1)*stride];
	dataRadius[radius_pos + 0]= cube[pos + ((size - 1)*stride)];
	// Apply boxcar filter to last data point
	for(i = filter_radius; i--;) { //desde el anterior en un radio. El mismo y los posteriores no son necesarios
		int iC = size + i - filter_radius - 1;
		if (iC >= 0){
			//data[(size - 1)*stride] += data[(iC)*stride];
			cube[pos + ((size - 1)*stride)] += cube[pos + ((iC)*stride)];
		}
	}
	//data[(size - 1)*stride] *= inv_filter_size;
	cube[pos + ((size - 1)*stride)] *= inv_filter_size;
	
	// Recursively apply boxcar filter to  all previous data points
	for(i = size - 1; i--;) {
		int iC1 = i - filter_radius;
		int iC2 = i + filter_radius + 1;
		float dataCpy1;
		float dataCpy2;

		//dataCpyArray[countdataCpyArray] = data[i*stride];
		dataRadius[radius_pos + countdataCpyArray] = cube[pos + (i*stride)];
		if (countdataCpyArray >= (int)filter_radius + 1){
			countdataCpyArray = 0;
		}else{
			countdataCpyArray++;
		}

		if (iC1 >= 0) {
			//dataCpy1 = data[iC1*stride];
			dataCpy1 = cube[pos + (iC1*stride)];
		}else{
			dataCpy1 = 0;
		}
		if (iC2 < (int)size) {
			//dataCpy2 = dataCpyArray[countdataCpyArray];
			dataCpy2 = dataRadius[radius_pos + countdataCpyArray];
		}else{
			dataCpy2 = 0;
		}
		//data[i*stride] = data[(i + 1)*stride] + (dataCpy1 - dataCpy2) * inv_filter_size;
		cube[pos+ (i*stride)] = cube[pos + ((i + 1)*stride)] + (dataCpy1 - dataCpy2) * inv_filter_size;
	}
	return;
}




























