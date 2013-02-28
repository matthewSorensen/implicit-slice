__device__ int sign(float x){
  if(x == 0.0){
    return 0;
  }else if(x < 0.0){
    return -1;
  }
  return 1;
}

__global__ void fast_firstpass(float* implicit,int* output, int width,int height){
  // We are going with 16*16 blocks, so we need (16 + 2) * 16 bytes per thread
  __shared__ char signs[288];
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = x + y * width;
  int replace = max(width,height);
  replace *= replace;

  if(width <= x || height <= y){ 
    _syncthreads();
    return;
  }

  int this = sign(implicit[i]);

  signs = &(signs[18 * blockIdx.y]);
  signs[blockIdx.x + 1] = this;

  if(x == 0){
    signs[0] = this;
  }else if(x == (width -1)){
    signs[x + 2] = this;
  }else if(blockId.x == 0){
    signs[0] = sgn(implicit[i-1]);
  } else if(blockId.x == (blockDim.x -1)){
    signs[17] = sgn(implicit[i+1]);
  }  

  _syncthreads();

  if(this == 1 && (signs[blockIdx.x] == -1 || signs[blockIdx.x + 2] == -1)){
    this = 0;
  }

  output[i] = this * replace;
}
