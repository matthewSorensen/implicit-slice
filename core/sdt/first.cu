

__device__ static int sgn(const float f){
  if(f == 0.0)
    return 0;
  if(signbit(f))
    return -1;
  return 1;
}

__device__ static float safe_copysign(float x, float y){
  // returns x with the sign of y
  if(x == 0.0 || y == 0.0){
    return 0.0
  }
  return copysign(x,y);
}

__device__ static float square(const x){
  return x * x;
}

__global__ void horizontal(float* sample, const int width, const int pitch, const int height){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(height <= x) return;
  sample = sample + x * pitch;

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
	  sample[x] = safe_copysign(fminf(square(x - upper), square(x - last)), sample[x]);
	}
      }
      last = i;
    } else if (ssgn == 0){
      last = i + 1;
    }
    psgn = ssgn;
  }
  if(last == -1){
    float val = safe_copysign(width * width, sample[0]);
    for(int i = 0; i < width; i++){
      sample[i] = val;
    }
  } else {
    for(int i = last; i < width; i++){
      sample[i] = safe_copysign(square(i - last), sample[i]);
    }
  }
}
