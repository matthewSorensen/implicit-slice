
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
  sample = &(sample[x]);
  
  int psgn = sgn(sample[0]);
  int last = -1;

  for(int i = 0; i < height; i++){
    int ssgn = sgn(sample[width * i]);

    if(ssgn != psgn){
      int upper = i;
      if(psgn == 0)
	upper--;   
      if(last == -1){
	for(int x = 0; x < height; x++){
	  sample[width * x] = safe_copysign(square(x - upper), sample[width * x]);
	}
      } else {
	for(int x = last; x <= height; x++){
	  sample[width * x] = safe_copysign(square(fminf(x - last,upper - x)), sample[width * x]);
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
    for(int i = 0; i < height; i++){
      sample[width * i] = val;
    }
  } else {
    for(int i = last; i < height; i++){
      sample[width * i] = safe_copysign(square(i - last), sample[width * i]);
    }
  }
  
}

__global__ void voxel_first_pass(float* sample, const int width, const int height){
 const int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(height <= x) return;
  sample = &(sample[x]);
  
  int psgn = sgn(sample[0]);
  int last = -1;

  for(int i = 0; i < height; i++){
    int ssgn = sgn(sample[width * i]);

    if(ssgn != psgn){
      int upper = i;
      if(last == -1){
	for(int x = 0; x < height; x++){
	  sample[width * x] = safe_copysign(square(x - upper), sample[width * x]);
	}
      } else {
	for(int x = last; x <= height; x++){
	  sample[width * x] = safe_copysign(square(fminf(x - last,upper - x)), sample[width * x]);
	}
      }
      last = upper;
    }
    psgn = ssgn;
  }
  if(last == -1){
    float val = safe_copysign(square(fmaxf(width, height) * 2), sample[0]);
    for(int i = 0; i < height; i++){
      sample[width * i] = val;
    }
  } else {
    for(int i = last; i < height; i++){
      sample[width * i] = safe_copysign(square(i - last), sample[width * i]);
    }
  }
}

#define INF 0x7f800000
#define NINF 0xff800000 


__global__ void second_pass(float* twod, float* bounds,int* verts, float* out,int width, int height){
  const int y = blockIdx.x * blockDim.x + threadIdx.x;
  if(height <= y) return;
  twod = &(twod[y * width]);
  out = &(out[y * width]);
  bounds = &(bounds[y * (width + 1)]);
  verts = &(verts[y * width]);
  
  int k = 0;
  bounds[0] = NINF;
  bounds[1] = INF;
  verts[0] = 0;
  
  for(int q = 1; q < width; q++){
    float sample = fabsf(twod[q]);
    float ss = sample + square(q);
    float inter = ss - fabsf(twod[verts[k]]) - square(verts[k]);
    
    inter *= 0.5 / (q - verts[k]);
    
    while(inter <= bounds[k]){
      verts[k] = 0;
      k--;
      inter = ss - fabsf(twod[verts[k]]) - square(verts[k]);
      inter *= 0.5 / (q - verts[k]);
    }
    
    k++;
    verts[k]  = q;
    bounds[k] = inter;
    bounds[k+1] = INF;
  }
  
  k = 0;
  for(int q = 0; q < width; q++){
    while(bounds[k+1] < q)
      k++;
    out[q] = square(q - verts[k]) + fabsf(twod[verts[k]]);
    }


}

__global__ void sign_and_sqrt(float* signs, float* values, float* output, int width, int height){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(width <= x || height <= y) return;
  int i = x + width * y;

  output[i] = safe_copysign(signs[i],sqrtf(values[i]));
}

