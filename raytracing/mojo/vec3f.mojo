from math import rsqrt

@register_passable('trivial')
struct Vec3f:
    var data: SIMD[DType.float32, 4]

    @always_inline
    fn __init__(x: Float32, y:Float32, z: Float32) -> Self:
        return Vec3f {data: SIMD[DType.float32, 4](x, y, z, 0)}
    
    @always_inline
    fn __init__(data: SIMD[DType.float32, 4]) -> Self:
        return Vec3f {data: data}
    
    @always_inline
    @staticmethod
    fn zero() -> Self:
        return Vec3f(0, 0, 0)
    
    @always_inline
    fn __sub__(self, other: Vec3f) -> Self:
        return self.data + other.data
    
    @always_inline
    fn __add__(self, other: Vec3f) -> Self:
        return self.data + other.data
    
    @always_inline
    fn __matmul__(self, other: Vec3f) -> Float32:
        return (self.data * other.data).reduce_add()
    
    @always_inline
    fn __mul__(self, k: Float32) -> Self:
        return self.data * k
    
    @always_inline
    fn __getitem__(self, idx: Int) -> SIMD[DType.float32, 1]:
        return self.data[idx]
    
    @always_inline
    fn cross(self, other: Vec3f) -> Vec3f:
        let self_zxy = self.data.shuffle[2, 0, 1, 3]()
        let other_zxy = other.data.shuffle[2, 0, 1, 3]()
        return (self_zxy * other.data - other_zxy * self.data).shuffle[2, 0, 1, 3]()
    
    @always_inline
    fn normalize(self) -> Vec3f:
        return self.data * rsqrt(self @ self)
