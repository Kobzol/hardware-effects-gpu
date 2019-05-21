## Shared memory resource limits
CUDA threads execute in group called warps, and warps in turn are formed into so-called blocks.
CUDA devices schedule blocks onto streaming multiprocessors (there are multiple of those on the device), when
a block is assigned to a SM, its warps can execute. The streaming multiprocessors however have limited
number of resources, one of these resources is the amount of available shared memory. This memory is shared
amongst all warps of a single block. The amount of memory that a block requires is either inferred by the compiler
from variable declarations (`__shared__ int sharedArray[1024]`) or it can be explicitly passed as a parameter when
launching a kernel (`kernel<<<gridDim, blockDim, sharedMemorySize, stream>>>(...)`);

A streaming multiprocessor can handle multiple blocks in parallel, however only if they do not take too many resources.
Suppose that a SM can handle 6 blocks and it has 96 KiB of shared memory. It will be able to run
6 blocks in parallel if they require `<= 96 KiB` of shared memory in total. But if each block requires for example
`48 KiB` of shared memory, the SM will be only able to execute 2 blocks in parallel, thus reducing the occupancy.

Note that even if the kernel doesn't use the shared memory at all or uses only a subset of it, if it's allocated it will
take up resources and it may prevent other blocks from being scheduled onto the SM. It is thus a design trade-off to
consider the size of the used shared memory versus the attainable occupancy of the streaming multiprocessors.

There are also other resource constraints that have to be taken into account to achieve optimal occupancy.
For example the number of registers available per thread or the total number of threads that can be handled concurrently
on a single streaming multiprocessor. For example if a SM can handle 6 blocks and 2048 threads in total, it will be able
to run 6 blocks with 128 threads each in parallel. But if each block has 1024 threads, then the SM will only be able
to run 2 blocks in parallel because of the thread limit.

Usage:
```bash
$ shared-memory-limit <sharedMemorySize>
```

The program will spawn 100 blocks for each SM. Each of the blocks will use `sharedMemorySize` bytes
of shared memory. The program will print the number of blocks executable in paralell by a SM and the amount of shared
memory that it has. If you set small amounts of shared memory, the blocks will be scheduled to the streaming multiprocessors
as quickly as possible, using up the available hardware. However, if you use too much shared memory for each block,
the SM will not be able to fill all of the available block spots because it will run out of shared memory space.

You may have to fiddle with the constants (block and grid dimensions and shared memory size) to observe slowdown on your
GPU. However you should be able to observe the achieved occupancy using `nvprof` in any case (if your GPU supports profiling):
```bash
$ nvprof --metrics achieved_occupancy ./shared-memory-limit/shared-memory-limit 1
Invocations                        Metric Name                        Metric Description         Min         Max         Avg
1000                        achieved_occupancy                        Achieved Occupancy    0.940840    0.951901    0.946277

$ nvprof --metrics achieved_occupancy ./shared-memory-limit/shared-memory-limit 32768
Invocations                        Metric Name                        Metric Description         Min         Max         Avg
1000                        achieved_occupancy                        Achieved Occupancy    0.360961    0.362433    0.361710
```

You can use the provided `benchmark.py` script to test various `sharedMemorySize` values
and plot their relative speeds.

```bash
$ python3 benchmark.py <path-to-executable>
```
