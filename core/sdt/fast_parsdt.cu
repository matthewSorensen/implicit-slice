#define MAXBLOCK 512

__device__ int square(int x){
  return x*x;
}

__global__ void fast_parsdt(int* samples, int width, const int height){

  __shared__ int coeffs[MAXBLOCK];
  __shared__ int verts[MAXBLOCK];

  // this requires that the thread size is (length of data,1)
  const int x = blockIdx.y * blockDim.y + threadIdx.y;
  const int y = blockIdx.x * blockDim.x + threadIdx.x;
  int* sample = &(samples[y + x * width]);
  int original = *sample;
  int out = abs(original);
 
  // Perform the first set of reductions
  coeffs[y] = out;
  __syncthreads();
  int otherindex = y ^ 1;
  int otherdata = coeffs[otherindex];

  if(out > otherdata){
    coeffs[y] = otherdata + 1;
    verts[y] = otherindex;
  }else{
    verts[y] = y;
  }
  
  __syncthreads();  
  
  int frame = y & ~1;
  int mask = 3; 

  while(width > 0){
    width >>= 1;

    int base = frame & ~3;
    int dest = base >> 1;

    int offset = base ^ frame;
    int half = offset >> 1;
    offset = offset | half;

    int par = y & mask;

    int low = square(y - verts[base + 1]) + coeffs[base + 1];
    int high = square(y - verts[base + 2]) + coeffs[base + 2];
    int extreme = square(y - verts[base + offset]) + coeffs[base + offset];

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
