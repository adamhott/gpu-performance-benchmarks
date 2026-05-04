#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace nbody_host {

struct Vec3 {
  float x, y, z;
};

constexpr int kMaxStack = 192;
constexpr float kG = 1.0f;

struct alignas(16) BHNode {
  float minx, miny, minz, maxx, maxy, maxz;
  float comx, comy, comz, mass;
  int body;
  int c0, c1, c2, c3, c4, c5, c6, c7;
};

struct Body {
  float x, y, z, m;
};

inline Vec3 make_v3(float x, float y, float z) { return Vec3{x, y, z}; }

inline int octant_from_box(float px, float py, float pz, float cx, float cy, float cz) {
  int q = 0;
  if (px >= cx) {
    q |= 1;
  }
  if (py >= cy) {
    q |= 2;
  }
  if (pz >= cz) {
    q |= 4;
  }
  return q;
}

inline void child_box(int oct, float minx, float miny, float minz, float maxx, float maxy,
                      float maxz, float* cminx, float* cminy, float* cminz, float* cmaxx,
                      float* cmaxy, float* cmaxz) {
  const float mx = 0.5f * (minx + maxx);
  const float my = 0.5f * (miny + maxy);
  const float mz = 0.5f * (minz + maxz);
  if (oct & 1) {
    *cminx = mx;
    *cmaxx = maxx;
  } else {
    *cminx = minx;
    *cmaxx = mx;
  }
  if (oct & 2) {
    *cminy = my;
    *cmaxy = maxy;
  } else {
    *cminy = miny;
    *cmaxy = my;
  }
  if (oct & 4) {
    *cminz = mz;
    *cmaxz = maxz;
  } else {
    *cminz = minz;
    *cmaxz = mz;
  }
}

inline float box_width(const BHNode& n) {
  return std::fmax(n.maxx - n.minx, std::fmax(n.maxy - n.miny, n.maxz - n.minz));
}

class BarnesHutCpu {
 public:
  explicit BarnesHutCpu(std::vector<Body> bodies_in) : bodies_(std::move(bodies_in)) {}

  std::vector<BHNode> build() {
    nodes_.clear();
    if (bodies_.empty()) {
      return nodes_;
    }
    nodes_.reserve(std::max(bodies_.size() * static_cast<std::size_t>(24),
                            static_cast<std::size_t>(4096)));
    float minx = bodies_[0].x;
    float miny = bodies_[0].y;
    float minz = bodies_[0].z;
    float maxx = bodies_[0].x;
    float maxy = bodies_[0].y;
    float maxz = bodies_[0].z;
    for (const Body& b : bodies_) {
      minx = std::fmin(minx, b.x);
      miny = std::fmin(miny, b.y);
      minz = std::fmin(minz, b.z);
      maxx = std::fmax(maxx, b.x);
      maxy = std::fmax(maxy, b.y);
      maxz = std::fmax(maxz, b.z);
    }
    const float pad = 0.05f * std::fmax(maxx - minx, std::fmax(maxy - miny, maxz - minz));
    minx -= pad;
    miny -= pad;
    minz -= pad;
    maxx += pad;
    maxy += pad;
    maxz += pad;

    const int root = alloc_empty_leaf(minx, miny, minz, maxx, maxy, maxz);
    for (std::size_t i = 0; i < bodies_.size(); ++i) {
      insert(static_cast<int>(i), root);
    }
    return nodes_;
  }

 private:
  std::vector<Body> bodies_;
  std::vector<BHNode> nodes_;

  int alloc_empty_leaf(float minx, float miny, float minz, float maxx, float maxy,
                       float maxz) {
    BHNode n{};
    n.minx = minx;
    n.miny = miny;
    n.minz = minz;
    n.maxx = maxx;
    n.maxy = maxy;
    n.maxz = maxz;
    n.comx = n.comy = n.comz = 0.f;
    n.mass = 0.f;
    n.body = -1;
    n.c0 = n.c1 = n.c2 = n.c3 = n.c4 = n.c5 = n.c6 = n.c7 = -1;
    nodes_.push_back(n);
    return static_cast<int>(nodes_.size() - 1);
  }

  void pull_aggregate(int idx) {
    BHNode& n = nodes_[idx];
    n.comx = n.comy = n.comz = 0.f;
    n.mass = 0.f;
    for (int k = 0; k < 8; ++k) {
      const int ch = child_idx(n, k);
      if (ch < 0) {
        continue;
      }
      const BHNode& c = nodes_[ch];
      n.mass += c.mass;
      n.comx += c.comx * c.mass;
      n.comy += c.comy * c.mass;
      n.comz += c.comz * c.mass;
    }
    if (n.mass > 0.f) {
      const float inv = 1.0f / n.mass;
      n.comx *= inv;
      n.comy *= inv;
      n.comz *= inv;
    }
  }

  static int child_idx(BHNode& n, int k) {
    switch (k) {
      case 0:
        return n.c0;
      case 1:
        return n.c1;
      case 2:
        return n.c2;
      case 3:
        return n.c3;
      case 4:
        return n.c4;
      case 5:
        return n.c5;
      case 6:
        return n.c6;
      default:
        return n.c7;
    }
  }

  static void set_child(BHNode& n, int k, int v) {
    switch (k) {
      case 0:
        n.c0 = v;
        break;
      case 1:
        n.c1 = v;
        break;
      case 2:
        n.c2 = v;
        break;
      case 3:
        n.c3 = v;
        break;
      case 4:
        n.c4 = v;
        break;
      case 5:
        n.c5 = v;
        break;
      case 6:
        n.c6 = v;
        break;
      default:
        n.c7 = v;
        break;
    }
  }

  void insert(int particle, int nodeIdx) {
    const Body& pb = bodies_[static_cast<std::size_t>(particle)];

    if (nodes_[nodeIdx].body >= 0) {
      const int old = nodes_[nodeIdx].body;
      nodes_[nodeIdx].body = -1;
      nodes_[nodeIdx].c0 = nodes_[nodeIdx].c1 = nodes_[nodeIdx].c2 = nodes_[nodeIdx].c3 =
          nodes_[nodeIdx].c4 = nodes_[nodeIdx].c5 = nodes_[nodeIdx].c6 = nodes_[nodeIdx].c7 = -1;
      const float cx = 0.5f * (nodes_[nodeIdx].minx + nodes_[nodeIdx].maxx);
      const float cy = 0.5f * (nodes_[nodeIdx].miny + nodes_[nodeIdx].maxy);
      const float cz = 0.5f * (nodes_[nodeIdx].minz + nodes_[nodeIdx].maxz);
      const int o_old = octant_from_box(bodies_[static_cast<std::size_t>(old)].x,
                                        bodies_[static_cast<std::size_t>(old)].y,
                                        bodies_[static_cast<std::size_t>(old)].z, cx, cy, cz);
      const int o_new = octant_from_box(pb.x, pb.y, pb.z, cx, cy, cz);
      float cminx, cminy, cminz, cmaxx, cmaxy, cmaxz;
      for (const int o : {o_old, o_new}) {
        if (child_idx(nodes_[nodeIdx], o) < 0) {
          child_box(o, nodes_[nodeIdx].minx, nodes_[nodeIdx].miny, nodes_[nodeIdx].minz,
                    nodes_[nodeIdx].maxx, nodes_[nodeIdx].maxy, nodes_[nodeIdx].maxz, &cminx,
                    &cminy, &cminz, &cmaxx, &cmaxy, &cmaxz);
          const int ch = alloc_empty_leaf(cminx, cminy, cminz, cmaxx, cmaxy, cmaxz);
          set_child(nodes_[nodeIdx], o, ch);
        }
      }
      insert(old, child_idx(nodes_[nodeIdx], o_old));
      insert(particle, child_idx(nodes_[nodeIdx], o_new));
      pull_aggregate(nodeIdx);
      return;
    }

    const bool has_child = (nodes_[nodeIdx].c0 >= 0) || (nodes_[nodeIdx].c1 >= 0) ||
                           (nodes_[nodeIdx].c2 >= 0) || (nodes_[nodeIdx].c3 >= 0) ||
                           (nodes_[nodeIdx].c4 >= 0) || (nodes_[nodeIdx].c5 >= 0) ||
                           (nodes_[nodeIdx].c6 >= 0) || (nodes_[nodeIdx].c7 >= 0);
    if (has_child) {
      const float cx = 0.5f * (nodes_[nodeIdx].minx + nodes_[nodeIdx].maxx);
      const float cy = 0.5f * (nodes_[nodeIdx].miny + nodes_[nodeIdx].maxy);
      const float cz = 0.5f * (nodes_[nodeIdx].minz + nodes_[nodeIdx].maxz);
      const int o = octant_from_box(pb.x, pb.y, pb.z, cx, cy, cz);
      int ch = child_idx(nodes_[nodeIdx], o);
      if (ch < 0) {
        float cminx, cminy, cminz, cmaxx, cmaxy, cmaxz;
        child_box(o, nodes_[nodeIdx].minx, nodes_[nodeIdx].miny, nodes_[nodeIdx].minz,
                  nodes_[nodeIdx].maxx, nodes_[nodeIdx].maxy, nodes_[nodeIdx].maxz, &cminx,
                  &cminy, &cminz, &cmaxx, &cmaxy, &cmaxz);
        ch = alloc_empty_leaf(cminx, cminy, cminz, cmaxx, cmaxy, cmaxz);
        set_child(nodes_[nodeIdx], o, ch);
      }
      insert(particle, ch);
      pull_aggregate(nodeIdx);
      return;
    }

    nodes_[nodeIdx].body = particle;
    nodes_[nodeIdx].mass = pb.m;
    nodes_[nodeIdx].comx = pb.x;
    nodes_[nodeIdx].comy = pb.y;
    nodes_[nodeIdx].comz = pb.z;
  }
};

inline int bh_child(const BHNode& n, int k) {
  switch (k) {
    case 0:
      return n.c0;
    case 1:
      return n.c1;
    case 2:
      return n.c2;
    case 3:
      return n.c3;
    case 4:
      return n.c4;
    case 5:
      return n.c5;
    case 6:
      return n.c6;
    default:
      return n.c7;
  }
}

inline void accel_accum_softened(Vec3 p, float ox, float oy, float oz, float mass, float eps2,
                                 Vec3* a) {
  float dx = ox - p.x;
  float dy = oy - p.y;
  float dz = oz - p.z;
  const float r2 = dx * dx + dy * dy + dz * dz + eps2;
  const float inv = 1.0f / std::sqrt(r2);
  const float inv2 = inv * inv;
  const float s = kG * mass * inv2 * inv;
  a->x += s * dx;
  a->y += s * dy;
  a->z += s * dz;
}

inline Vec3 bh_accel_one(const std::vector<BHNode>& nodes, const std::vector<Body>& bodies,
                         int i, int root, float theta, float eps2) {
  const Vec3 p = make_v3(bodies[static_cast<std::size_t>(i)].x, bodies[static_cast<std::size_t>(i)].y,
                         bodies[static_cast<std::size_t>(i)].z);
  Vec3 a = make_v3(0.f, 0.f, 0.f);
  int stack[kMaxStack];
  int sp = 0;
  stack[sp++] = root;

  while (sp > 0) {
    const int ni = stack[--sp];
    const BHNode& n = nodes[static_cast<std::size_t>(ni)];
    const bool is_leaf = (n.body >= 0);

    if (is_leaf) {
      if (n.body == i) {
        continue;
      }
      const Body& ob = bodies[static_cast<std::size_t>(n.body)];
      accel_accum_softened(p, ob.x, ob.y, ob.z, ob.m, eps2, &a);
      continue;
    }

    float dx = n.comx - p.x;
    float dy = n.comy - p.y;
    float dz = n.comz - p.z;
    const float dist = std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
    const float s = std::fmax(std::fmax(n.maxx - n.minx, n.maxy - n.miny), n.maxz - n.minz);
    const bool open = (s / dist) >= theta;

    if (!open) {
      if (n.mass > 0.f) {
        accel_accum_softened(p, n.comx, n.comy, n.comz, n.mass, eps2, &a);
      }
    } else {
      for (int k = 7; k >= 0; --k) {
        const int ch = bh_child(n, k);
        if (ch >= 0) {
          if (sp < kMaxStack) {
            stack[sp++] = ch;
          }
        }
      }
    }
  }
  return a;
}

inline void bh_accel_all(const std::vector<BHNode>& nodes, const std::vector<Body>& bodies,
                         int nBodies, int root, float theta, float eps2, std::vector<Vec3>* accels) {
  accels->resize(static_cast<std::size_t>(nBodies));
  for (int i = 0; i < nBodies; ++i) {
    (*accels)[static_cast<std::size_t>(i)] =
        bh_accel_one(nodes, bodies, i, root, theta, eps2);
  }
}

inline void direct_accel_cpu(const std::vector<Body>& bodies, float eps2, std::vector<Vec3>* out) {
  const int n = static_cast<int>(bodies.size());
  out->assign(static_cast<std::size_t>(n), make_v3(0.f, 0.f, 0.f));
  for (int i = 0; i < n; ++i) {
    const Vec3 pi = make_v3(bodies[static_cast<std::size_t>(i)].x, bodies[static_cast<std::size_t>(i)].y,
                            bodies[static_cast<std::size_t>(i)].z);
    for (int j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }
      const Body& bj = bodies[static_cast<std::size_t>(j)];
      float dx = bj.x - pi.x;
      float dy = bj.y - pi.y;
      float dz = bj.z - pi.z;
      const float r2 = dx * dx + dy * dy + dz * dz + eps2;
      const float inv = 1.0f / std::sqrt(r2);
      const float inv2 = inv * inv;
      const float s = kG * bj.m * inv2 * inv;
      (*out)[static_cast<std::size_t>(i)].x += s * dx;
      (*out)[static_cast<std::size_t>(i)].y += s * dy;
      (*out)[static_cast<std::size_t>(i)].z += s * dz;
    }
  }
}

inline float acc_mag(const Vec3& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline void require_accel_match(const std::vector<Vec3>& ref, const std::vector<Vec3>& got,
                                float rel_tol, const char* ctx) {
  if (ref.size() != got.size()) {
    std::fprintf(stderr, "%s: size mismatch\n", ctx);
    std::exit(EXIT_FAILURE);
  }
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float mr = acc_mag(ref[i]);
    const float mg = acc_mag(got[i]);
    const float scale = std::fmax(1.0f, std::fmax(mr, mg));
    const float ex = std::fabs(ref[i].x - got[i].x);
    const float ey = std::fabs(ref[i].y - got[i].y);
    const float ez = std::fabs(ref[i].z - got[i].z);
    if (ex > rel_tol * scale || ey > rel_tol * scale || ez > rel_tol * scale) {
      std::fprintf(stderr,
                   "%s: accel mismatch at %zu ref=(%.6g,%.6g,%.6g) got=(%.6g,%.6g,%.6g)\n", ctx, i,
                   static_cast<double>(ref[i].x), static_cast<double>(ref[i].y),
                   static_cast<double>(ref[i].z), static_cast<double>(got[i].x),
                   static_cast<double>(got[i].y), static_cast<double>(got[i].z));
      std::exit(EXIT_FAILURE);
    }
  }
}

inline double bh_flops_estimate(double n) {
  if (n <= 1.0) {
    return 0.0;
  }
  const double lg = std::log2(n);
  return 80.0 * n * lg;
}

inline double bh_bytes_estimate(double n) {
  return static_cast<double>(n) * (sizeof(Body) + sizeof(Vec3) + 64.0);
}

}  // namespace nbody_host
