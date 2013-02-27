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
    int low = min(square(y - verts[frame]) + coeffs[frame], square(y - verts[frame +1]) + coeffs[frame +1]);
    int high = min(square(y - verts[frame + 2]) + coeffs[frame + 2], square(y - verts[frame +3]) + coeffs[frame +3]);

    out = min(out, min(low,high));
    int dest = (frame>>1) & ~1;
      
    __syncthreads();

    int par = y & mask;

    if(par == 0){
      int otherv = verts[frame + 2];
      int otherc = coeffs[frame + 2];
      if((square(y - otherv) + otherc) < low){
	coeffs[dest] = otherc;
	verts[dest] = otherv;
      }else{
	coeffs[dest] = coeffs[frame];
	verts[dest] = verts[frame];
      }
    }else if(par == mask){
      int otherv = verts[frame - 1];
      int otherc = coeffs[frame - 1];
      if((square(y - otherv) + otherc) < high){
	coeffs[dest + 1] = otherc;
	verts[dest + 1] = otherv;
      }else{
	coeffs[dest + 1] = coeffs[frame+1];
	verts[dest + 1] = verts[frame+1];
      }
    }

    __syncthreads();

    frame = dest;
    mask = (mask << 1) + 1;
    
  }
   
  *sample = out;  
}
