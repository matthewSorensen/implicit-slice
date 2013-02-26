__device__ int sgn(const float f){
  if(f == 0.0)
    return 0;
  if(f < 0.0){
    return -1;
  }
  return 1;
  //  if(signbit(f))
  //   return -1;
  //return 1;
}

__device__ float safe_copysign(const float x, const float y){
  // returns x with the sign of y
  if(x == 0.0 || y == 0.0){
    return 0.0;
  }
  return copysignf(x,y);
}

__device__ float square(const float x){
  return x * x;
}

__global__ void implicit_first_pass(float* sample, const int width, const int height){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(height <= x) return;
  sample = &(sample[x * width]);
  
  int psgn = sgn(sample[0]);
  int last = -1;

  for(int i = 0; i < width; i++){
    int ssgn = sgn(sample[i]);

    if(ssgn != psgn){
      int upper = i;
      if(psgn == 0)
	upper--;   
      if(last == -1){
	for(int x = 0; x < upper; x++){
	  sample[x] = safe_copysign(square(x - upper), sample[x]);
	}
      } else {
	for(int x = last; x <= upper; x++){
	  sample[x] = safe_copysign(square(fminf(x - last,upper - x)), sample[x]);
	}
      }
      last = upper;
    } else if (ssgn == 0){
      last = i + 1;
    }
    psgn = ssgn;
  }
  if(last == -1){
    float val = safe_copysign(square(fmaxf(width, height) * 2), sample[0]);
    for(int i = 0; i < width; i++){
      sample[i] = val;
    }
  } else {
    for(int i = last; i < width; i++){
      sample[i] = safe_copysign(square(i - last), sample[i]);
    }
  }
  
}

__global__ void voxel_first_pass(float* sample, const int width, const int height){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(height <= x) return;
  sample = &(sample[x * width]);

  int psgn = sgn(sample[0]);
  int last = -1;

  for(int i = 0; i < width; i++){
    int ssgn = sgn(sample[i]);

    if(ssgn != psgn){
      int upper = i;
     
      if(last == -1){
	for(int x = 0; x < upper; x++){
	  sample[x] = safe_copysign(square(x - upper), sample[x]);
	}
      } else {
	for(int x = last; x <= upper; x++){
	  sample[x] = safe_copysign(square(fminf(x - last,upper - x)), sample[x]);
	}
      }
      last = upper;
    }
    psgn = ssgn;
  }
  if(last == -1){
    float val = safe_copysign(square(fmaxf(width, height) * 2), sample[0]);
    for(int i = 0; i < width; i++){
      sample[i] = val;
    }
  } else {
    for(int i = last; i < width; i++){
      sample[i] = safe_copysign(square(i - last), sample[i]);
    }
  }
}

__global__ void second_pass(float* bounds,int* verts, float* out,int width, int height){

}


//    second_pass(bounds,verts,drv.Out(sample), width, height, block = block, grid = (bls, 1))


// sampled implicit => one-d signed, squared => rebind as texture (can we eliminate all copies?) => compute unsigned, squared


__global__ void sign_and_sqrt(float* values, float* output, int width, int height){
}

