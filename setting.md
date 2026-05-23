### Set-up pixi files
```
[workspace]
channels = ["conda-forge"]
name = "langsplat"
description = "Baseline of language field_LangSplat"
platforms = ["linux-aarch64"]
version = "0.1.0"

[system-requirements]
cuda = "12.6"

[dependencies]
plyfile = "0.8.1.*"
python = "3.9.*"
pip = "*"
setuptools = "*"
wheel = "*"
ninja = "*"
numpy = "<2"
tqdm = "*"
opencv = "*"
tensorboard = "*"
jaxtyping = "*"
matplotlib = "*"
typing = "*"
pathlib = "*"
pillow = "*"
cuda-version = "12.6.*"
cuda-nvcc = "12.6.*"
pytorch = { version = "2.6.*", build = "cuda126*" }
torchvision = { version = "0.21.*", build = "cuda126*" }

[pypi-dependencies]
open-clip-torch = "*"
segment-anything = { path = "submodules/segment-anything-langsplat", editable = true }

[tasks]
install-render-ext = "CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST='9.0' python -m pip install --no-build-isolation submodules/simple-knn submodules/langsplat-rasterization"
check-render-ext = "python -c \"import simple_knn._C; import diff_gaussian_rasterization; print('render extensions ok')\""

```
### Installing submodules
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

### File management in miyabi

SAM checkpoint, .pixi, pixi cache, dataset are all in work directory


