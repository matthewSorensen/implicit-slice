#define MAXBLOCK 512

__device__ int square(int x){
  return x*x;
}

__global__ void fast_parsdt(const int* samples, const int width, int height){

  __shared__ int coeffs[MAXBLOCK];
  __shared__ int verts[MAXBLOCK];

  // this requires that the thread size is (1,length of data)
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int* sample = &(samples[x + y * width]);
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
  
  while(height > 0){
    height >>= 1;
  
    int low = square(y - verts[j]) + coeffs[j];
    int high = square(y - verts[j+1]) + coeffs[j+1];

    out = min(out, min(lower,higher));
    int dest = (frame>>1) & ~1;
      
    __syncthreads();

    int par = y & mask;

    if(par == 0){
      int otherv = verts[frame + 2];
      int otherc = coeffs[frame + 2];
      if(square(y - otherv) + otherc < low){
	coeffs[dest] = otherc;
	verts[dest] = otherv;
      }else{
	coeffs[dest] = coeffs[j];
	verts[dest] = verts[j];
      }
    }else if(par == mask){
      int otherv = verts[frame - 1];
      int otherc = coeffs[frame - 1];
      if(square(y - otherv) + otherc < high){
	coeffs[dest + 1] = otherc;
	verts[dest + 1] = otherv;
      }else{
	coeffs[dest + 1] = coeffs[j+1];
	verts[dest + 1] = verts[j+1];
      }
    }

    __syncthreads();

    frame = dest;
    mask = (mask << 1) + 1;
    
  }
   
  *sample = out;
}
