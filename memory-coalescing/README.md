## Memory access coalescing
Threads on CUDA devices are grouped in so-called warps (size of a warp is usually 32 threads).
Each of these threads can initiate a global memory request (load/store) in the same cycle.
Spamming the DRAM with 32 memory requests is not optimal, therefore the memory controller tries
to combine concurrent accesses falling inside the same 32 or 128 byte block into a single
memory transaction (this is known as memory access coalescing). Threads in a warp should thus
attempt to access as few of those memory blocks as possible, to minimize the number of 
memory transactions and thus maximize the memory bus utilization.

Older CUDA architectures used 128 byte blocks and loaded data from global memory through the
L1 cache. On newer CUDA architectures, global memory loads by default bypass the L1 cache and
use 32 byte regions for coalescing. This means that if the threads access a lot of independent blocks, less
bandwidth might be wasted (since only 32 bytes will be loaded instead of 128 for each block).
The used access type (L1+128B or L2+32B) can be either configured globally as a compiler flag
or it can be selected on a per-access basis using PTX
(https://docs.nvidia.com/cuda/kepler-tuning-guide/index.html#l1-cache).

It is also important for the blocks to be properly aligned. If the concurrent accesses of a warp
fit into 32 bytes, but they are not aligned and span the block boundary
(for example one access at address `28` and another at address `36`), two memory transactions will
be generated instead of just one. Using CUDA memory allocation functions should always return
memory that is aligned to (at least) 256 bytes
(https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses).

The memory controller is smart and can coalesce loads/stores even if the individual threads
access memory in a "shuffled" order (e.g. thread 0 accesses address `P + 4`, thread 1 accesses `P`).
On prehistoric CUDA architectures (compute capability < 2), this was apparently not the case.

You can see graphical examples of memory coalescing in this presentation:
http://developer.download.nvidia.com/CUDA/training/bandwidthlimitedkernels_webinar.pdf

Usage:
```bash
$ memory-coalescing <startOffset> <moveOffset>
```

The program will spawn exactly one warp of threads (32 threads in a single block). The threads will
then repeatedly read and write a 32-bit integer to global memory at address starting
at `threadIdx.x * startOffset`. The address will then be incremented by `32 * moveOffset` before the next access.

If you select `startOffset=1` and `moveOffset=32`, then all threads of the warp will access consecutive
memory locations at the same time. Thread 0 will access address `P`, thread 1 will access address
`P + 4` etc. After that each thread will move 32 integers forward to skip the addresses already accessed
by the other threads, so thread 0 will access `P + 128`, thread 1 will access `P + 128 + 4` etc.
Since all concurrent accesses fall into a single 128B aligned block, it will generate the fewest
possible amount of memory transactions (1 or 4, depending on the access type described above).

If you select `startOffset=32` and `moveOffset=1`, then thread 0 will access locations `P`, `P + 4`,
`P + 8`, thread 1 will access locations `P + 32`, `P + 36` etc. This would probably be the logical
distribution on a CPU (to avoid false sharing and use the cache better), however on a CUDA GPU it
is not optimal. Since each thread access will hit a separate memory block, 32 memory transactions
will be generated for every access.

We can verify this using `nvprof`, using metrics that calculate the number of global memory transactions.

Case with `startOffset=1` and `moveOffset=32`:
```bash
$ nvprof --metrics gld_transactions_per_request,gst_transactions_per_request,gld_transactions,gst_transactions ./memory-coalescing 1 32
Invocations                        Metric Name                        Metric Description         Min         Max         Avg
1000              gld_transactions_per_request      Global Load Transactions Per Request    4.000000    4.000000    4.000000
1000              gst_transactions_per_request     Global Store Transactions Per Request    4.000000    4.000000    4.000000
1000                          gld_transactions                  Global Load Transactions        4000        4000        4000
1000                          gst_transactions                 Global Store Transactions        4000        4000        4000
```

In this optimal case, exactly 4 memory transactions are generated on each access (since we access 32 * 4 = 128 bytes),
which corresponds to four 32 byte memory blocks.

Case with `startOffset=32` and `moveOffset=1`:
```bash
$ nvprof --metrics gld_transactions_per_request,gst_transactions_per_request,gld_transactions,gst_transactions ./memory-coalescing 32 1
Invocations                        Metric Name                        Metric Description         Min         Max         Avg
1000              gld_transactions_per_request      Global Load Transactions Per Request   32.000000   32.000000   32.000000
1000              gst_transactions_per_request     Global Store Transactions Per Request   32.000000   32.000000   32.000000
1000                          gld_transactions                  Global Load Transactions       32000       32000       32000
1000                          gst_transactions                 Global Store Transactions       32000       32000       32000
```

We can see that there are 32 memory transactions for each memory access, which wastes the memory bandwidth and you should
observe a slowdown in this case.

As stated above, the accesses of the individual threads may be shuffled. Therefore if you for example reverse the access
pattern like this:

`int threadId = 31 - threadIdx.x;`

so that thread 0 will access the last element of the block and thread 31 the first one, it should generate the same amount
of memory transactions (in the case of `startOffset=1` and `moveOffset=32`).

However misaligning the accesses still presents a challenge for the memory controller. If you add this line to the beginning
of the kernel:
`memory = memory + 1;`
all of the accesses will be misaligned by 4 bytes.

Misaligned accesses with `startOffset=1` and `moveOffset=32`:
```bash
$ nvprof --metrics gld_transactions_per_request,gst_transactions_per_request,gld_transactions,gst_transactions ./memory-coalescing 1 32
Invocations                        Metric Name                        Metric Description         Min         Max         Avg
1000              gld_transactions_per_request      Global Load Transactions Per Request    4.970500    4.970500    4.970500
1000              gst_transactions_per_request     Global Store Transactions Per Request    5.000000    5.000000    5.000000
1000                          gld_transactions                  Global Load Transactions       19882       19882       19882
1000                          gst_transactions                 Global Store Transactions       20000       20000       20000
```

The last integer of each access falls into a fifth memory block, therefore there are 5 memory transactions per access
instead of just four.

```bash
$ python3 benchmark.py <path-to-executable>
```
