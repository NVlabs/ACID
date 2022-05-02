try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'src.utils.libmcubes.mcubes',
    sources=[
        'src/utils/libmcubes/mcubes.pyx',
        'src/utils/libmcubes/pywrapper.cpp',
        'src/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'src.utils.libmise.mise',
    sources=[
        'src/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'src.utils.libsimplify.simplify_mesh',
    sources=[
        'src/utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)


# Gather all extension modules
ext_modules = [
    mcubes_module,
    mise_module,
    simplify_mesh_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    }
)
