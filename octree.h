#ifdef SOP_FP_DOUBLE
#define FLOAT double
#define FLOAT3 double3
#else
#define FLOAT float
#define FLOAT3 float3
#endif

const int nbody = 76;

extern FLOAT3* unc_pos;
extern FLOAT3* dev_buffer1;
extern FLOAT3* dev_buffer2;

class Octree_node {
  int children_index;
  FLOAT3 m_center;
  int begin, end;
  float m_width;

public:
  __host__ __device__ Octree_node() : begin(0), end(0), m_width(0), children_index(-1) {}

  __host__ __device__ __forceinline__ FLOAT3 center() const {
    return m_center;
  }

  __host__ __device__ __forceinline__ void set_center(float xi, float yi, float zi){
    m_center.x = xi; m_center.y = yi; m_center.z = zi;
  }

  __host__ __device__ __forceinline__ int num_points(){
    return end - begin;
  }

  __host__ __device__ __forceinline__ int points_begin(){
    return begin;
  }

  __host__ __device__ __forceinline__ int points_end(){
    return end;
  }

  __host__ __device__ __forceinline__ void set_range(int ibegin, int iend){
    begin = ibegin; end = iend;
  }

  __host__ __device__ __forceinline__ float width(){
    return m_width;
  }

  __host__ __device__ __forceinline__ void set_width(float width){
    m_width = width;
  }

  __host__ __device__ __forceinline__ int children(){
    return children_index;
  }

  __host__ __device__ __forceinline__ void set_children(int children){
    children_index = children;
  }

};

struct Parameters {
  int point_selector;
  int depth;
  int min_points;
  int max_depth;

  __host__ __device__ Parameters(int idepth) : point_selector(0), depth(0), min_points(1), max_depth(idepth) {}

  __host__ __device__ Parameters(const Parameters &params, bool):
  point_selector((params.point_selector+1)%2),
  depth(params.depth+1) {}
};

