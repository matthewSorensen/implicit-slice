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

__global__ void extract_zeros(float* implicit,int* output, int width,int height){
  // We are going with 16*16 blocks, so we need (16 + 2) * 16 bytes per thread
  __shared__ char allsigns[288];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = x + y * width;
  int replace = max(width,height);
  replace *= replace;

  if(width <= x || height <= y){ 
    __syncthreads();
    return;
  }

  int here = sgn(implicit[i]);

  char* signs = &(allsigns[18 * threadIdx.y]);
  signs[threadIdx.x + 1] = here;
  
  if(x == 0){
    signs[0] = here;
  }else if(x == (width -1)){
    signs[threadIdx.x + 2] = here;
  }else if(threadIdx.x == 0){
    signs[0] = sgn(implicit[i-1]);
  } else if(threadIdx.x == (blockDim.x -1)){
    signs[17] = sgn(implicit[i+1]);
  }  
  __syncthreads();
  if((here == 1) && ((signs[threadIdx.x] == -1) || (signs[threadIdx.x + 2] == -1))){
    here = 0;
  }
  output[i] = here * replace;
}

#define MAXBLOCK 512

__global__ void edt_pass(int* samples, int width, const int height){

  __shared__ int coeffs[MAXBLOCK];
  __shared__ int verts[MAXBLOCK];

  // this requires that the thread size is (length of data,1)
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  int* sample = &(samples[x + y * width]);
  int original = *sample;
  int out = abs(original);
 
  // Perform the first set of reductions
  coeffs[x] = out;
  __syncthreads();
  int otherindex = x ^ 1;
  int otherdata = coeffs[otherindex];

  if(out > otherdata){
    coeffs[x] = otherdata + 1;
    verts[x] = otherindex;
  }else{
    verts[x] = x;
  }
  
  __syncthreads();  
  
  int frame = x & ~1;
  int mask = 3; 

  while(width > 0){
    width >>= 1;

    int base = frame & ~3;
    int dest = base >> 1;

    int offset = base ^ frame;
    int half = offset >> 1;
    offset = offset | half;

    int par = x & mask;

    int low = square(x - verts[base + 1]) + coeffs[base + 1];
    int high = square(x - verts[base + 2]) + coeffs[base + 2];
    int extreme = square(x - verts[base + offset]) + coeffs[base + offset];

    out = min(out,min(high,min(low,extreme)));
    
    if(par == 0 || par == mask){
      if(high < extreme || low < extreme){
	offset = (offset + 2) & 3;
      }
      
      int vertex = verts[base + offset];
      int coefficient = coeffs[base + offset];
      
      __syncthreads();
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
  
  if(original < 0) out = -1 * out;
  
  *sample = out;  
}
