#define MAXBLOCK 512

__device__ int sgn(float x){
  if(x == 0.0){
    return 0;
  }else if(x < 0.0){
    return -1;
  }
  return 1;
}

__device__ int square(int x){
  return x*x;
}

__global__ void binarize(int* output, int2 size, float* implicit){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = x + y * size.x;
  const int replace = square(max(size.x,size.y));
  
  if(size.x <= x || size.y <= y) return;
  
  output[i] = sgn(implicit[i]) * replace;
}

__global__ void edt_pass(int* samples, const int width, const int height, const int dim){

  __shared__ int coeffs[MAXBLOCK];
  __shared__ int verts[MAXBLOCK];
  __shared__ int signs[MAXBLOCK];

  // this requires that the thread size is (length of data,1)
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  int* sample = &(samples[x + y * width]);

  int frame;
  int pos;
  int size;
  if(dim){
    frame = y & ~1;
    pos = y;
    size = height;
  }else{
    frame = x & ~1;
    pos = x;
    size = width;
  }
 
  // Perform the first set of reductions
  int out = *sample;
  int sign = sgn(out);
  
  out = abs(out);
  signs[pos] = sign * pos;
  coeffs[pos] = out;

  __syncthreads();
  int otherindex = pos ^ 1;
  int otherdata = coeffs[otherindex];
  int othersign = signs[otherindex];

  if(othersign * sign < 0){
    if(sign == -1){
      out = 0;
      coeffs[pos] = 0;
      verts[pos] = pos;
      signs[pos] = 0;
    }else{
      out = 1;
      coeffs[pos] = 1;
      verts[pos] = otherindex;
    }
  }else if(out > otherdata){
    coeffs[pos] = otherdata + 1;
    verts[pos] = otherindex;
  }else{
    verts[pos] = pos;
  }
  
  __syncthreads();  


  int mask = 3; 

  while(size > 0){
    size >>= 1;

    int base = frame & ~3;
    int dest = base >> 1;

    int offset = base ^ frame;
    int half = offset >> 1;
    offset = offset | half;

    int par = pos & mask;

    int lowvertex = verts[base + 1];
    int highvertex = verts[base + 2];

    int lowcoeff = coeffs[base + 1];
    int highcoeff = coeffs[base + 2];

    int lowsgn = signs[base+1];
    int highsgn = signs[base+2];

    if((0 > lowsgn * highsgn) && (size > 1)){
      lowvertex = abs(min(lowsgn,highsgn));
      lowcoeff = 0;
    }

    int low = square(pos - lowvertex) + lowcoeff;
    int high = square(pos - highvertex) + highcoeff;
    int extreme = square(pos - verts[base + offset]) + coeffs[base + offset];

    out = min(out,min(high,min(low,extreme)));
    
    if(par == 0 || par == mask){
      int vertex;
      int coefficient;

      if(high < extreme || low < extreme){
	if(high < low){
	  vertex = highvertex;
	  coefficient = highcoeff;
	}else{
	  vertex = lowvertex;
	  coefficient = lowcoeff;
	}
      } else {
	vertex = verts[base + offset];
	coefficient = coeffs[base + offset];
      }
      
      int s = signs[base + 3 * half];

      __syncthreads();
      signs[dest + half] = s;
      coeffs[dest + half] = coefficient;
      verts[dest + half] = vertex;
      __syncthreads();
    }else{
      __syncthreads();
      __syncthreads();
    }
    
    frame = dest;
    mask = (mask << 1) + 1;    
  }
  
  *sample = sign * out;  
}



__global__ void signed_sqrt(int* values, int2 size, float* output){ // int width, int height){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(size.x <= x || size.y <= y) return;
  const int i = x + size.x * y;

  float out = 1.0;
  int value = values[i];
  
  if(value < 0){
    out = -1.0;
    value = -1 * value;
  }

  output[i] = out * sqrtf((float) value) / ((float) min(size.y,size.x));
}

