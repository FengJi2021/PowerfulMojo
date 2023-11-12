from python import Python
from python.object import PythonObject
from vec3f import Vec3f


struct Image:
    # reference count
    var rc: Pointer[Int]
    # image is array with dimension vector3 * width * height
    var pixels: Pointer[Vec3f]
    var height: Int
    var width: Int

    fn __init__(inout self, height: Int, width: Int):
        self.height = height
        self.width = width
        self.pixels = Pointer[Vec3f].alloc(height * width)
        self.rc = Pointer[Int].alloc(1)
        self.rc.store(1)
    
    fn __copyinit__(inout self, other: Self):
        other._inc_rc()
        self.rc = other.rc
        self.pixels = other.pixels
        self.width = other.width
        self.height = other.height
    
    fn __del__(owned self):
        self._dec_rc()
    
    fn _get_rc(self) -> Int:
        return self.rc.load()
    
    fn _inc_rc(self):
        let rc = self._get_rc()
        self.rc.store(rc + 1)
    
    fn _dec_rc(self):
        let rc = self._get_rc()
        if rc > 1:
            self.rc.store(rc - 1)
            return
        self._free()
    
    fn _free(self):
        self.rc.free()
        self.pixels.free()
    
    @always_inline
    fn set(self, row: Int, col: Int, value: Vec3f) -> None:
        self.pixels.store(self._pos_to_index(row, col), value)
    
    fn _pos_to_index(self, row: Int, col: Int) -> Int:
        return row * self.width + col
    
    def to_numpy_image(self) -> PythonObject:
        let np = Python.import_module("numpy")
        let plt = Python.import_module("matplotlib.pyplot")

        let np_image = np.zeros((self.height, self.width, 3), np.float32)

        # saving image using 
        let out_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type=__mlir_type[`!kgen.pointer<scalar<f32>>`]
            ](
                SIMD[DType.index, 1](
                    np_image.__array_interface__["data"][0].__index__()
                ).value
            )
        )

        let in_pointer = Pointer(
            __mlir_op.`pop.index_to_pointer`[
                _type=__mlir_type[`!kgen.pointer<scalar<f32>>`]
            ](
                SIMD[DType.index, 1](
                    self.pixels.__as_index()
                ).value
            )
        )

        for raw in range(self.height):
            for col in range(self.width):
                let index = self._pos_to_index(raw, col)
                for dim in range(3):
                    out_pointer.store(
                        index * 3 + dim,
                        in_pointer[index * 4 + dim]
                    )
        
        return np_image

def load_image(fname: String) -> Image:
    let np = Python.import_module("numpy")
    let plt = Python.import_module("matplotlib.pyplot")

    let np_image = plt.imread(fname)
    let rows = np_image.shape[0].__index__()
    let cols = np_image.shape[1].__index__()
    let image = Image(rows, cols)

    let in_pointer = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type=__mlir_type[`!kgen.pointer<scalar<f32>>`]
        ](
            SIMD[DType.index, 1](
                np_image.__array_interface__["data"][0].__index__()
            ).value
        )
    )
    let out_pointer = Pointer(
        __mlir_op.`pop.index_to_pointer`[
            _type=__mlir_type[`!kgen.pointer<scalar<f32>>`]
        ](SIMD[DType.index, 1](image.pixels.__as_index()).value)
    )

    for row in range(rows):
        for col in range(cols):
            let index = image._pos_to_index(row, col)
            for dim in range(3):
                out_pointer.store(
                    index * 4 + dim, in_pointer[index * 3 + dim]
                )
    return image

def render(image: Image):
    np = Python.import_module("numpy")
    plt = Python.import_module("matplotlib.pyplot")
    colors = Python.import_module("matplotlib.colors")
    dpi = 32
    fig = plt.figure(
        1,
        [image.width // 10, image.height // 10],
        dpi
    )

    plt.imshow(image.to_numpy_image())
    plt.axis("off")
    plt.show()

fn main():
    let image = Image(192, 256)

    for row in range(image.height):
        for col in range(image.width):
            image.set(
                row,
                col,
                Vec3f(Float32(row) / image.height, Float32(col) / image.width, 0),
            )
    try:
        render(image)
    except:
        print("Error: matplotlib is not installed")