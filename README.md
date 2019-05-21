# GPU hardware effects
This repository demonstrates hardware effects that can affect application performance on
Nvidia CUDA GPUs. For each effect I try to create a proof of concept program that is as small
as possible so that it can be understood easily. My goal is to demonstrate effects that are caused
by the underlying hardware architecture design and cannot be explained by looking at the
source code alone.

Related repository with CPU hardware effects: https://github.com/kobzol/hardware-effects

The demonstrated effects of course depend heavily on your GPU microarchitecture and model.
Right now there the example programs are focused on CUDA GPUs.

Currently the following effects are demonstrated:

- bank conflicts
- memory access coalescing

Every example directory has a README that explains the individual effects.

Isolating those hardware effects can be very tricky, so it's possible that some of the
examples are actually demonstrating something entirely else (or nothing at all :) ).
If you have a better explanation of what is happening, please let me know in the issues.

### Build
```bash
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make -j
```

If you want to use the benchmark scripts (written in Python 3), you should
also install the Python dependencies:
```bash
$ pip install -r requirements.txt
```

### Docker
You will need the Nvidia Docker runtime to use CUDA inside Docker containers:
```bash
$ sudo apt-get install nvidia-docker2
```

Build the image:
```bash
$ docker build -t hardware-effects-gpu .
```

Then run it:
```bash
# interactive run
$ docker run --runtime=nvidia --rm -it hardware-effects-gpu

# directly launch a program
$ docker run --runtime=nvidia hardware-effects-gpu build/bank-conflicts/bank-conflicts 1
```

### License
MIT

### Resources
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
