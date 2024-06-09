from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension , CppExtension

setup(
    name='cpu',
    ext_modules=[
        CppExtension(
            name='cpu',
            # sources=['cpu.cpp'],
            sources=['cpu.cpp', '/Users/lzx/Desktop/TInyMPC/car_simulation/src/tinympc/admm.cpp'],
            include_dirs=['/Users/lzx/Desktop/TInyMPC/car_simulation/include/Eigen',
            '/Users/lzx/Desktop/TInyMPC/car_simulation/examples',
            '/Users/lzx/Desktop/TInyMPC/car_simulation/src'],  
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)