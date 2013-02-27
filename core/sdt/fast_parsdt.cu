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

    int llv = verts[base];
    int llc = coeffs[base];
    int lhv = verts[base+1];
    int lhc = coeffs[base+1];

    int hlv = verts[base+2];
    int hlc = coeffs[base+2];
    int hhv = verts[base+3];
    int hhc = coeffs[base+3];
   
    int low  = min(square(y - llv) + llc,square(y - lhv) + lhc);
    int high = min(square(y - hhv) + hhc,square(y - hlv) + hlc);

    out = min(out, min(low,high));
    int dest = (frame>>1) & ~1;
     
    int par = y & mask;

    __syncthreads();

    if(par == 0){
      if((square(y - llv ) + llc) < square(y - hlv) + hlc){
	coeffs[dest] = llc;
	verts[dest] = llv;
      }else{
	coeffs[dest] = hlc;
	verts[dest] = hlv;
      }
    }else if(par == mask){
      if(square(y - hhv) + hhc < square(y - lhv) + lhc){
	coeffs[dest+1] = hhc;
	verts[dest+1] = hhv;
      }else{
	coeffs[dest+1] = lhc;
	verts[dest+1] = lhv;
      }
    }

    __syncthreads();

    frame = dest;
    mask = (mask << 1) + 1;
    
  }
   
  *sample = out;  
}
