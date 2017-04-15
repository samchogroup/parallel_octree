#include <cstdlib>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>
#include "octree.h"

#define COORD_MAX 40.0f

__host__ static __inline__ thrust::tuple<float, float, float> rnd()
{
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> dist(-COORD_MAX, COORD_MAX);
  thrust::tuple<float, float, float> t(dist(rng), dist(rng), dist(rng));
  return t;
}

__device__ bool check_points(Octree_node &node, Points *points, int num_points, Parameters params){
  if(params.depth >= params.max_depth || num_points <= params.min_points){
    if(params.point_selector == 1){
      int it = node.points_begin(); int end = node.points_end();
      for(it += threadIdx.x; it < end; it += blockDim.x){
        points[0].set_point(it, points[1].get_point(it));
      }
    }
    return true;
  }
  return false;
}

__device__ void count_points(const Points &in_points, int *smem, int range_begin, int range_end, float3 center){
  if(threadIdx.x < 8) smem[threadIdx.x] = 0;
  __syncthreads();

  for(int iter=range_begin+threadIdx.x; iter<range_end; iter+=blockDim.x){
    float3 p = in_points.get_point(iter);

    int x = p.x < center.x ? 0 : 1;
    int y = p.y < center.y ? 0 : 1;
    int z = p.z < center.z ? 0 : 1;

    int i = x*4 + y*2 + z;

    atomicAdd(&smem[i], 1);
  }
  __syncthreads();
}

__device__ void scan_offsets(int node_points_begin, int* smem){
  int *smem2 = &smem[8];
  if(threadIdx.x == 0){
    for(int i = 0; i < 8; i++){
      smem2[i] = i == 0 ? 0 : smem2[i-1] + smem[i-1];
    }
    for (int i = 0; i < 8; i++){
      smem2[i] += node_points_begin;
    }
  }
  __syncthreads();
}

__device__ void reorder_points(Points &out_points, const Points &in_points, int *smem, int range_begin, int range_end, float3 center){
  int *smem2 = &smem[8];

  for(int iter = range_begin+threadIdx.x; iter<range_end; iter+=blockDim.x){
    float3 p = in_points.get_point(iter);

    int x = p.x < center.x ? 0 : 1;
    int y = p.y < center.y ? 0 : 1;
    int z = p.z < center.z ? 0 : 1;

    int i = x*4 + y*2 + z;

    int dest = atomicAdd(&smem2[i], 1);
    out_points.set_point(dest, p);
  }

  __syncthreads();
}

__device__ void prepare_children(Octree_node *children, Octree_node &node, int *smem){

  int child_offset = 8*node.id();

  for(int i = 0; i < 8; i++){
    children[child_offset+i].set_id(8*node.id()+(i*8));
  }

  const float3 center = node.center();
  float half = node.width() / 2.0f;
  float quarter = half / 2.0f;

  for(int i = 0; i < 8; i++){
    float xf, yf, zf;
    xf = i / 4 == 0 ? -1.0f : 1.0f;
    yf = (i-4) / 4 == 0 ? -1.0f : 1.0f;
    zf = i % 2 == 0 ? -1.0f : 1.0f;

    children[child_offset+i].set_center(center.x + quarter * xf,
                                        center.y + quarter * yf,
                                        center.z + quarter * zf);

    children[child_offset+i].set_width(half);
    children[child_offset+i].set_range(smem[8+i], smem[i]+smem[8+i]);
  }
}

__global__ void build_octree_kernel(Octree_node *nodes, Points *points, Parameters params){
  __shared__ int smem[16];

  Octree_node &node = nodes[blockIdx.x];
  node.set_id(node.id() + blockIdx.x);
  int num_points = node.num_points();

  bool exit = check_points(node, points, num_points, params);
  if(exit) return;

  float3 center = node.center();

  int range_begin = node.points_begin();
  int range_end = node.points_end();
  const Points &in_points = points[params.point_selector];
  Points &out_points = points[(params.point_selector + 1) % 2];

  count_points(in_points, smem, range_begin, range_end, center);

  scan_offsets(node.points_begin(), smem);

  reorder_points(out_points, in_points, smem, range_begin, range_end, center);

  if(threadIdx.x == blockDim.x-1){
    Octree_node *children = &nodes[params.nodes_in_level];
    prepare_children(children, node, smem);
    build_octree_kernel<<<8, blockDim.x, 16*sizeof(int)>>>(children, points, Parameters(params, true));
  }
}

int main(){
  const int nbody = 76;

  thrust::device_vector<float> x_d0(nbody);
  thrust::device_vector<float> x_d1(nbody);
  thrust::device_vector<float> y_d0(nbody);
  thrust::device_vector<float> y_d1(nbody);
  thrust::device_vector<float> z_d0(nbody);
  thrust::device_vector<float> z_d1(nbody);

  // RandGen rnd;
  thrust::generate(
    thrust::make_zip_iterator(thrust::make_tuple(x_d0.begin(), y_d0.begin(), z_d0.begin())),
    thrust::make_zip_iterator(thrust::make_tuple(x_d1.end(), y_d1.end(), z_d1.end())),
    rnd
  );

  Points points_init[2];
  points_init[0].set(thrust::raw_pointer_cast(&x_d0[0]),
                    thrust::raw_pointer_cast(&y_d0[0]),
                    thrust::raw_pointer_cast(&z_d0[0]));
  points_init[1].set(thrust::raw_pointer_cast(&x_d1[0]),
                    thrust::raw_pointer_cast(&y_d1[0]),
                    thrust::raw_pointer_cast(&z_d1[0]));

  Points *points;
  cudaMalloc((void**) &points, 2*sizeof(Points));
  cudaMemcpy(points, points_init, 2*sizeof(Points), cudaMemcpyHostToDevice);

  Octree_node root;
  root.set_range(0, nbody);
  root.set_width(2*COORD_MAX);
  Octree_node *nodes;
  cudaMalloc((void **)&nodes, nbody*sizeof(Octree_node));
  cudaMemcpy(nodes, &root, sizeof(Octree_node), cudaMemcpyHostToDevice);

  Parameters params(nbody);
  const int THREADS_PER_BLOCK = 128;
  const int shared_mem = 16*sizeof(int);
  build_octree_kernel<<<1, THREADS_PER_BLOCK, shared_mem>>>(nodes, points, params);
  cudaGetLastError();

  cudaFree(nodes);
  cudaFree(points);

}
