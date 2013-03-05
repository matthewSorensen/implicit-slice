__device__ int sgn(float x){
  if(x == 0.0){
    return 0;
  }else if(x < 0.0){
    return -1;
  }
  return 1;
}

__global__ void fast_firstpass(float* implicit,int* output, int width,int height){
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

  char* signs = &(allsigns[18 * blockIdx.y]);
  signs[blockIdx.x + 1] = here;

  if(x == 0){
    signs[0] = here;
  }else if(x == (width -1)){
    signs[x + 2] = here;
  }else if(blockIdx.x == 0){
    signs[0] = sgn(implicit[i-1]);
  } else if(blockIdx.x == (blockDim.x -1)){
    signs[17] = sgn(implicit[i+1]);
  }  

  __syncthreads();

  if(here == 1 && (signs[blockIdx.x] == -1 || signs[blockIdx.x + 2] == -1)){
    here = 0;
  }

  output[i] = here * replace;
}
