## Set-up pixi files
```
[workspace]
# Prefer conda-forge packages so the environment resolves consistently.
channels = ["conda-forge"]
name = "langsplat"
description = "Baseline of language field_LangSplat"
# MIYABI GH200 compute nodes are Linux ARM64.
platforms = ["linux-aarch64"]
version = "0.1.0"

[system-requirements]
# Match the CUDA version available on the GH200 nodes.
cuda = "12.6"

[dependencies]
# Reads and writes Gaussian point-cloud PLY files.
plyfile = "0.8.1.*"
# Python 3.9 is compatible with the LangSplat code and CUDA extensions here.
python = "3.9.*"
# Needed to build the local CUDA extension submodules below.
pip = "*"
setuptools = "*"
wheel = "*"
ninja = "*"
# Keep compatibility with PyTorch and packages expecting the NumPy 1.x API.
numpy = "<2"
tqdm = "*"
opencv = "*"
# Records training metrics for TensorBoard logging in train.py.
tensorboard = "*"
jaxtyping = "*"
matplotlib = "*"
typing = "*"
pathlib = "*"
pillow = "*"
# Provide the CUDA compiler/toolkit used to compile rasterization extensions.
cuda-version = "12.6.*"
cuda-nvcc = "12.6.*"
# Use a CUDA-enabled PyTorch build matching CUDA 12.6.
pytorch = { version = "2.6.*", build = "cuda126*" }
torchvision = { version = "0.21.*", build = "cuda126*" }

[pypi-dependencies]
# Produces CLIP language features in preprocess.py.
open-clip-torch = "*"
# Install the LangSplat-compatible SAM fork directly from the submodule.
segment-anything = { path = "submodules/segment-anything-langsplat", editable = true }

[tasks]
# Compile the custom KNN and Gaussian rasterization CUDA operators on a GPU node.
install-render-ext = "CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST='9.0' python -m pip install --no-build-isolation submodules/simple-knn submodules/langsplat-rasterization"
# Confirm that both compiled CUDA operators can be imported.
check-render-ext = "python -c \"import simple_knn._C; import diff_gaussian_rasterization; print('render extensions ok')\""

```
## Installing submodules
```
git submodule add https://github.com/minghanqin/segment-anything-langsplat submodules/segment-anything-langsplat
git submodule add https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
git submodule add https://github.com/minghanqin/langsplat-rasterization submodules/langsplat-rasterization
git submodule update --init --recursive
```

Create the Pixi environment once after cloning or changing dependencies.
On a login node, mock the CUDA virtual package because the GPU is not visible there.

```bash
CONDA_OVERRIDE_CUDA=12.6 pixi install
```

Build the CUDA render extensions once inside a GH200 GPU job.
Run this again only after recreating the Pixi environment or changing PyTorch, CUDA, or extension sources.

```bash
pixi run install-render-ext
pixi run check-render-ext
```

## File management in miyabi

```text
/home/p47004/
`-- base_langsplat/                 # Git-tracked source code and settings
    |-- arguments/
    |-- autoencoder/
    |-- eval/
    |-- gaussian_renderer/
    |-- scene/
    |-- utils/
    |-- preprocess.py
    |-- render.py
    |-- train.py
    |-- pixi.toml
    |-- pixi.lock
    |-- ckpt -> work directory      # symlink, ignored by Git
    |-- data -> work directory      # symlink, ignored by Git
    |-- output -> work directory    # symlink, ignored by Git
    `-- .pixi -> work directory     # symlink, ignored by Git

/work/gp47/p47004/
|-- pixi/
|   `-- base_langsplat/
|       `-- .pixi/                  # Pixi environment
`-- langsplat/
    |-- ckpt/                       # SAM model
    |   `-- sam_vit_h_4b8939.pth
    |
    |-- data/                       # Dataset and generated training inputs
    |   `-- lerf/
    |       |-- figurines/
    |       |   |-- images/
    |       |   |-- sparse/0/
    |       |   |-- language_features/       # generated 512-d features
    |       |   `-- language_features_dim3/  # generated 3-d features
    |       |-- ramen/
    |       |-- teatime/
    |       |-- waldo_kitchen/
    |       `-- label/
    |
    `-- output/                     # trained models
        |-- sofa/
        |   |-- sofa_1/
        |   |-- sofa_2/
        |   `-- sofa_3/
        `-- teatime/
            |-- teatime_1/
            |-- teatime_2/
            `-- teatime_3/
```

- `data/` contains source images, COLMAP data, and generated language
  features used as training input.
- `output/` contains trained LangSplat model outputs.
