from pathlib import Path
from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

curope_dir = "splatweaver/model/encoder/backbone/croco/curope"
all_cuda_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()

setup(
    ext_modules=[
        CUDAExtension(
            name="splatweaver.model.encoder.backbone.croco.curope.curope",
            sources=[
                f"{curope_dir}/curope.cpp",
                f"{curope_dir}/kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)