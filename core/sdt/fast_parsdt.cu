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
  int out = *sample;
 
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
    int dest = (frame>>1) & ~1; 
    int par = y & mask;

    int lowest = square(y - verts[base]) + coeffs[base];
    int low = square(y - verts[base + 1]) + coeffs[base + 1];
    int high = square(y - verts[base + 2]) + coeffs[base + 2];
    int highest = square(y - verts[base + 3]) + coeffs[base + 3];

    out = min(out,min(min(lowest,low),min(high,highest)));

    int vertex = 0;
    int coefficient = 0;

    if(par == 0){
      if(high < lowest){
	vertex = verts[base + 2];
	coefficient = coeffs[base + 2];
      }else{
	vertex = verts[base];
	coefficient = coeffs[base];
      }
    }else if (par == mask){
      if(low <  highest){
	vertex = verts[base + 1];
	coefficient = coeffs[base + 1];
      }else{
	vertex = verts[base + 3];
	coefficient = coeffs[base + 3];
      }
      par = 1;
    }else{
      par = 2;
    }
    __syncthreads();
    if(par < 2){
      coeffs[dest + par] = coefficient;
      verts[dest + par] = vertex;
    }
    __syncthreads();

    frame = dest;
    mask = (mask << 1) + 1;    
  }
   
  *sample = out;  
}
