class Points {
  float *x;
  float *y;
  float *z;

public:
  __host__ __device__ Points(): x(NULL), y(NULL), z(NULL) {}

  __host__ __device__ Points(float *xi, float *yi, float *zi): x(xi), y(yi), z(zi) {}

  __host__ __device__ __forceinline__ float3 get_point(int idx) const {
    return make_float3(x[idx], y[idx], z[idx]);
  }

  __host__ __device__ __forceinline__ void set_point(int idx, const float3 &p){
    x[idx] = p.x; y[idx] = p.y; z[idx] = p.z;
  }

  __host__ __device__ __forceinline__ void set(float *xi, float *yi, float *zi){
    x = xi; y = yi; z = zi;
  }
};

class Octree_node {
  int m_id;
  float3 m_center;
  int begin, end;
  float m_width;

public:
  __host__ __device__ Octree_node() : m_id(0), begin(0), end(0), m_width(0) {}

  __host__ __device__ int id() const {
    return m_id;
  }

  __host__ __device__ void set_id(int new_id){
    m_id = new_id;
  }

  __host__ __device__ __forceinline__ float3 center() const {
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

};

struct Parameters {
  int point_selector;
  int nodes_in_level;
  int depth;
  int min_points;
  int max_depth;

  __host__ __device__ Parameters(int idepth) : point_selector(0), nodes_in_level(1), depth(0), min_points(1), max_depth(idepth) {}

  __host__ __device__ Parameters(const Parameters &params, bool):
  point_selector((params.point_selector+1)%2),
  nodes_in_level(8*params.nodes_in_level),
  depth(params.depth+1) {}
};
