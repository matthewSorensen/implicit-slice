__global__ void tile(float* out, int2 out_size, float* pattern, int2 pat_size, int2 offset){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(out_size.x <= x || out_size.y <= y){
    return;
  }

  const int i = x + out_size.x * y;
  const int j = ((x + offset.x) % pat_size.x) + (((y + offset.y) % pat_size.y) * pat_size.x);

  out[i] = pattern[j];
}

__global__ void fuse(float* out,  int2 dim, float* model, float offset){
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int i = x + y * dim.x;
  if(dim.x <= x || dim.y <= y){
    return;
  }

  float m = model[i] + offset;

  if(m > 0)
    out[i] = m;
}
