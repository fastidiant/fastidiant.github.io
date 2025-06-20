<!-- ---
layout: post
title:  "Optimizing HGEMM Using Tensor Cores"
date:   2025-03-04 08:52:08 -0600
categories: jekyll update
--- -->


{% include mathjax.html %}

* TOC
{:toc}

# Introduction
This article documents my recent endeavor to create an optimized matrix multiplication kernel in CUDA that utilizes tensor cores on an NVIDIA Tesla T4 GPU. The objective is to compute the equation $D = \alpha * A * B + \beta * C$ with the highest possible speed. In this formula, $D, A, B$, and $C$ represent large matrices containing half-precision floating-point numbers, while $\alpha$ and $\beta$ are scalar constants. This specific problem is commonly identified as a **H**alf-precision **Ge**neralized **M**atrix **M**ultiply, or **HGEMM** for short.

My interest in tensor cores was recently sparked for two main reasons. Firstly, it appears that [most](https://www.semianalysis.com/i/136469751/the-gpu-rich) generative AI training and inference activities today are conducted on A100 and H100 GPUs. Secondly, nearly all of this training and inference work almost certainly relies on the tensor cores within these devices, as they provide a significant throughput advantage for matrix mathematics compared to not using them. A quote from [this source](https://hazyresearch.stanford.edu/blog/2024-05-12-tk) illustrates this point:
>An H100 GPU has 989 TFLOPs of half-precision matrix multiply compute, and ~60 TFLOPs of “everything else”. So, every cycle the tensor core is in use, you’re getting at least 94% utilization of the hardware. And every cycle the tensor core is not in use, you’re getting no more than 6% utilization of the hardware.

Considering their immense importance in today's world, it struck me when I began this project that there is a disproportionately small amount of information and discussion available online about their direct usage. I soon discovered that this scarcity of online dialogue is likely because developing algorithms that employ them is a somewhat specialized interest. While the fundamental mechanics of invoking them are not difficult, crafting a kernel that can harness them to anywhere near their full capability *is* a challenge. Their tremendous throughput demands that to utilize them effectively, you must transfer bytes through the GPU's memory hierarchy with maximum efficiency and overlap this data movement with computation. There are specific algorithmic methods that one must employ to get the full value from their tensor cores, and this article serves as a deep dive into these methods.

I uncovered the implementation specifics primarily by exploring the NVIDIA [CUTLASS](https://github.com/NVIDIA/cutlass/tree/main) forums and its source code. I wrote this piece to ensure my own comprehension of the material and also with the hope that it might assist fellow GPU enthusiasts who are trying to work with tensor cores. It is important to note that this entire project was executed on a Turing architecture GPU, which represented the state of the art in 2018. Although none of the performance challenges discussed are exclusive to this architecture, some of the optimization strategies are. An interesting observation I made during this work is that the Hopper architecture includes dedicated hardware support that directly mitigates some of the performance issues and bottlenecks I encountered while optimizing for an older GPU. This suggests that more modern GPUs warrant their higher price not just through increased floating-point throughput, but also with features that reduce the cognitive load for programmers aiming to optimize kernels for them.

When I embarked on this project, my objective was to develop a kernel with performance comparable to the cuBLAS [hgemm](https://docs.nvidia.com/cuda/cublas/#cublas-level-3-function-reference) implementation, which is NVIDIA's closed-source, gold-standard solution. I progressively optimized a sequence of six kernels, where the [first](https://github.com/alexarmbr/matmul-playground/blob/main/src/kernel1.cu) achieved a mere 8% of the cuBLAS throughput, while the [final](https://github.com/alexarmbr/matmul-playground/blob/main/src/kernel6.cu) version reached a respectable 96% of the cuBLAS throughput for 8192x8192 matrices.

This post includes a background section that covers some theoretical concepts that are useful to keep in mind when considering how to optimize matrix-based kernels. The remainder of the article details the six algorithmic techniques I employed to make my kernel execute as quickly as possible.

Below is a table presenting the performance comparison for all of the kernels:
![table6](/images/table6.png)

# Background
## The memory wall

In the approximately 70 years since humanity began constructing transistor-based computers, the ability to perform arithmetic has grown at a Moore's Law exponential rate, whereas the capacity to move data from its storage location to the computation unit has not experienced similar exponential growth. This disparity is known as the [memory wall](https://en.wikipedia.org/wiki/Random-access_memory#Memory_wall), and it stands as one of the foremost challenges in computer architecture today, [particularly](https://horace.io/brrr_intro.html) for deep learning workloads and most especially for tensor core algorithms. For us, this implies that if we aim to harness the ~65 trillion FLOPs per second our tensor cores can deliver, transferring the necessary volume of bytes per second from DRAM could prove to be a significant obstacle.

## Roofline charts
The [roofline](https://en.wikipedia.org/wiki/Roofline_model) model provides a framework for analyzing this dilemma with greater precision. The core concept involves envisioning a simplified computer featuring a two-level memory hierarchy: fast memory and slow memory. We can only execute computations on data residing in fast memory, at a peak rate denoted by $\tau$ FLOP/sec. The slow memory possesses unlimited capacity and can transfer data to the fast memory at a rate of $\beta$ bytes/sec. Due to the memory wall phenomenon, $\tau$ is substantially larger than $\beta$.

![simple_computer](/images/simple_computer.png)

Any particular computation involves a specific number of FLOPs that must be executed; for instance, multiplying an $M$ by $K$ matrix with a $K$ by $N$ matrix requires $2 * M * N * K$ FLOPs. The more FLOP/sec our algorithm achieves, the more rapidly we can complete the matrix multiplication. The roofline model establishes an upper limit on the achievable FLOP/sec, constrained by $\tau$ and $\beta$, which are fixed characteristics of our hardware. We will denote the achieved FLOP/sec as $T$ for throughput, and its upper bound as $T_{max}$.

The maximum achievable FLOP/sec ($T_{max}$) is represented as a function of a variable known as *computational intensity*, or $I$ for short, which is a property of the algorithm we design. This metric quantifies the "data reuse" of our algorithm in units of FLOP/byte: for every byte moved from slow memory to fast memory, it measures how many FLOPs are performed on it. According to the roofline model, an algorithm designer's primary goal is to create an algorithm with high computational intensity, which means maximizing $I$. In practical terms, this involves moving a segment of data from slow to fast memory and then executing as many useful operations on it as the specific algorithm allows. Reusing data in fast memory is crucial for performance because our memory bandwidth $\beta$ is limited; it is a small value compared to $\tau$, making the transfer of this data chunk from slow to fast memory an expensive operation. We maximize its value by performing as many useful FLOPs on it as we can.

The roofline model states that the upper bound on FLOP/sec ($T_{max}$) we can attain is the lesser of our computational intensity multiplied by memory bandwidth, and the peak floating-point throughput of our hardware.

$$ T_{max}=min(\beta * I, \tau) $$

This model suggests that $T_{max}$ can be constrained in two ways:
- $T_{max}$ can never be greater than $\tau$. Even if we were to perform an infinite number of operations on each byte transferred into fast memory, we are ultimately limited by the hardware's peak floating-point throughput. The value of $\tau$ is typically enormous; for the T4 GPU, for example, $\tau$ is 65,000,000,000,000 FLOP/second. If $\tau$ is our constraint, we are in a favorable position; this situation is described as being *compute bound*.
- However, $T_{max}$ can also be constrained by the device's memory bandwidth multiplied by the algorithm's computational intensity. If $\tau$ were infinite, the achieved floating-point throughput would simply be the number of bytes/sec moved into fast memory times the number of FLOPs performed per byte moved, which is $\beta * I$ (note how multiplying $\beta * I$ results in units of FLOP/sec). If $\beta * I$ is smaller than $\tau$, this term becomes the limiting factor on $T_{max}$, a scenario known as being *memory bound*. The course of action in this situation is to revise your algorithm to increase $I$, with the aim of making it compute bound.

Here is a visual representation of the entire concept; observe how we can shift from being memory bound to compute bound by adjusting $I$:
![roofline](/images/roofline.png)

The red dotted line in this diagram is known as the "balance point" of the hardware; it represents the level of arithmetic intensity, in units of (FLOP/byte), that we must exceed to transition from being memory bound to compute bound. If we label this value as $I^*$, then it follows that $I^* * \beta=\tau$, or equivalently $I^*=\frac{\tau}{\beta}$. This is a characteristic of a specific computer, defined as its peak floating-point throughput divided by its memory bandwidth. Due to Moore's law, arithmetic throughput has advanced much more rapidly than memory bandwidth, which has resulted in a general trend where newer computers have higher balance points.

## Rooflines for the NVIDIA Tesla T4
By substituting some numbers specific to the GPU we are using, we can examine the resulting roofline to inform our algorithm design and gain perspective on what to expect. On a real computer, there isn't just one $\tau$ and $\beta$; instead, there are multiple hardware instructions, each with a distinct peak throughput $\tau$, and various types of memory, each with a different bandwidth $\beta$.

### Tensor Core vs. FFMA
I found it beneficial at first to compare the tensor cores' balance point with the balance point for the standard single-precision math units, both in relation to global memory. This roofline analysis offers some insight into why creating an efficient kernel is more demanding when using tensor core instructions compared to more conventional, less specialized math instructions.

First, we must determine the global memory bandwidth, $\beta_{gmem}$, of our device. NVIDIA's specification sheets provide a *theoretical* memory bandwidth, which is [practically never](https://forums.developer.nvidia.com/t/theoretical-bandwidth-vs-effective-bandwidth/48005/3?u=a14armbr) attainable. The actual figure can be determined through a benchmark; a whitepaper [here](https://arxiv.org/pdf/1903.07486) indicates that the T4's achievable memory bandwidth is 220 GB/sec (which is 68% of the 320 GB/sec theoretical memory bandwidth).

Next, we need to find the peak floating-point throughput both with and without the tensor core. Similar to memory, the theoretical figures are [not genuinely achievable](https://www.thonking.ai/p/strangely-matrix-multiplications) without the GPU overheating or melting. I consider it reasonable to use the measured throughput of the cuBLAS half-precision (which uses tensor cores) and single-precision (which does not use tensor cores) GEMM kernels as the achievable floating-point throughput values. Examining the assembly of the cuBLAS half-precision kernel reveals that the main work is performed by `HMMA.1688`, an instruction that executes a small, hardware-accelerated matrix multiplication (more details on this later). For the single-precision GEMM kernel, the instruction responsible for the work is called `FFMA`, which is a scalar fused multiply-accumulate operation, $d=a*b+c$. Based on my benchmarks, the tensor core HMMA.1688 throughput is 49,439 GFLOP/sec, which we will designate as $\tau_{HMMA}$. The non-tensor core FFMA throughput is 7,455 GFLOP/sec, which we will refer to as $\tau_{FFMA}$. These values represent 76% and 92% of their respective theoretical peak throughputs, which appears plausible. The resulting rooflines are depicted as follows (these plots are usually shown on a log/log scale, but this one is not):

![t4_roofline](/images/t4_roofline.png)

This plot should offer some intuition regarding the relative difficulty of writing a kernel that reaches peak FLOP/sec with tensor core instructions versus one that does so with fused multiply-add instructions. The difficulty arises from the fact that achieving a throughput of $\tau_{HMMA}$ requires approximately 6.6 times more arithmetic intensity than is needed to reach $\tau_{FFMA}$. The two balance points in this chart indicate that with FFMA instructions, we can execute about 33 FLOPs in the time it takes a byte to move from global memory, while with tensor cores, we can perform 224 FLOPs in the same duration. This implies that if we took a kernel that achieved the maximum flops possible with FFMA instructions, merely substituting the fused multiply-adds in the inner loop with tensor core instructions would be *insufficient* to attain high tensor core utilization. We would also need to enhance the data movement code to boost the computational intensity by a factor of six. This is one of the aspects that makes writing a tensor core GEMM particularly interesting!

### Shared memory vs. L2 cache vs. global memory
If our aim is to develop a kernel that can effectively utilize the tensor cores, we must be mindful of our computer's memory hierarchy. The roofline model simplifies the memory hierarchy into two storage types: one large and slow, and the other small and instantaneous. In reality, there are more than two levels, with each having different bandwidth and capacity, as well as distinct considerations that must be taken into account to enable efficient access.

![t4_memory_hierarchy](/images/t4_memory_hierarchy.png)

In this era of the memory wall, effectively using the faster and smaller levels of the memory hierarchy is paramount. This demands some cleverness due to their limited size; for example, on the T4, the on-chip shared memory offers 16.6 times the bandwidth of global memory, but on any given streaming multiprocessor (or SM), it can only hold 64 KiB. If we are dealing with large matrices, this is only enough capacity for a very small fraction of the problem.

![t4_memory_roofline](/images/t4_memory_roofline.png)

The plot contrasts the balance point of tensor cores with respect to:
- global memory or DRAM, which is the largest and slowest tier of the memory hierarchy.
- the L2 cache, which holds recently used data from DRAM and is shared among the 16 SMs on the T4.
- shared memory, which is a per-SM, high-speed memory that is managed explicitly.

Global memory possesses a balance point of 224, which signifies that if all our memory accesses target DRAM, we must execute 224 FLOPs for each byte read from DRAM to keep our tensor cores occupied. This proves to be a very difficult requirement, as we will see later when we analyze how our algorithm's parameters influence the balance point (a sneak peek reveals that achieving this balance point would be counterproductive given the T4's fast memory capacity and other performance factors). Fortunately, the L2 cache provides a solution, as its balance point relative to tensor cores is 38, a much more attainable figure. If a significant portion of our memory accesses can hit the L2 cache instead of traveling all the way to global memory, we stand a good chance of being compute-bound rather than memory-bound. The takeaway from this is that the L2 cache is essential for us.

Shared memory functions as an explicitly managed cache that stores small segments of the input matrices locally to a specific SM (an SM is somewhat analogous to a single CPU core). Within the SM, threads load their own local part of the problem from shared memory into register memory, which is where data must be located to be processed. When shared memory operates at its maximum bandwidth, its balance point with respect to the tensor core is 13, meaning we need to cache enough data in registers to perform 13 FLOPs for every byte read from shared memory. It turns out that each SM has sufficient register memory to make this readily achievable. As we optimize this part of the algorithm, the main challenge will be to enable shared memory to function at its full bandwidth, which in practice involves organizing the data layout to avoid bank conflicts during reads and writes. Once shared memory reaches its full bandwidth, achieving adequate arithmetic intensity will be straightforward. I believe the shared memory balance point of 13 is worth noting, however, as it indicates that shared memory by itself is not fast enough to reach peak tensor core throughput. The lesson here is that we require registers.

## Theoretical arithmetic intensity
Modern computers generally exhibit an imbalance between their arithmetic throughput and memory bandwidth; as a result, kernels that perform a high volume of arithmetic relative to data movement make more effective use of the hardware. At this stage, we must consider the algorithm we are implementing and momentarily set aside hardware considerations.

### Matrix Multiplication vs Matrix Addition

Any given algorithm has a maximum possible arithmetic intensity, and our objective as algorithm designers is to create a kernel that attains an arithmetic intensity as near to this upper limit as we can. A comparison between the maximum achievable arithmetic intensity when adding two $N$ by $N$ matrices versus multiplying them highlights how different algorithms have different upper bounds in this respect.

![multiplication_vs_addition](/images/multiplication_vs_addition.png)

For matrix addition, calculating a single output element needs just one arithmetic operation, which means that the amount of data movement and computation will always be directly proportional when we execute this algorithm. When adding two $N$x$N$ matrices, the volume of data is $O(N^2)$, and the amount of computation needed is also $O(N^2)$. Consequently, the ratio of compute to data is $\frac{O(N^2)}{O(N^2)}=O(1)$, indicating that matrix addition will likely be memory-bound on any modern hardware, regardless of how cleverly we design the algorithm. There simply is not much math required relative to the amount of data movement, so the upper limit on achievable arithmetic intensity is low. Many operations in deep learning fall into this low arithmetic intensity category, where a technique known as kernel fusion can be beneficial.

Matrix multiplication, however, is not fated to be memory-bound, because it demands more arithmetic relative to its problem size. When multiplying two $N$ by $N$ matrices, the data volume is also $O(N^2)$, but the required computation is $O(N^3)$ ($O(N)$ operations per output element, multiplied by $O(N^2)$ output elements). This makes the ratio of compute to data $\frac{O(N^3)}{O(N^2)}=O(N)$. There is a factor of $N$ more computation needed than data movement. The upper limit on the arithmetic intensity we can reach grows with the matrix dimension $N$. If we are multiplying matrices that are large enough, we should be able to design an algorithm with sufficient arithmetic intensity to be compute-bound instead of memory-bound.

In short, the arithmetic intensity we attain is determined by the kernel we create, and it must be less than or equal to an upper limit set by the algorithm our kernel implements. The achieved arithmetic intensity, in conjunction with our machine's parameters $\tau$ and $\beta$, dictates whether we are memory-bound or compute-bound. If our algorithm's maximum arithmetic intensity permits it, our goal is to optimize our kernel until it becomes compute-bound rather than memory-bound.

## Achievable arithmetic intensity on a simple computer
For the multiplication of two $N$ by $N$ matrices, the highest possible arithmetic intensity we can reach is $O(N)$. The question now is, how do we apply all of this when it's time to actually write a kernel? To address this question, we require a model of the computer we are running on; to begin, we will use the simple computer with its fast and slow memory.

### worst case
The initial implementation of multiplication between two N x N matrices ($C=A*B$) on our simple computer appears as follows. We fetch each value just as it is needed and save each output as soon as its calculation is complete. What is the ratio of computation to data movement in this case? Is it anywhere near the ideal of $O(N)$?
```
allocate registers a,b,c in fast memory
for i=1...N:
    for j=1...N:
        c = 0
        for k=1...N:
            load A(i,k) into a
            load B(k,j) into b
            c += a * b
        store c into C(i,j) 

```
My conceptual model of this implementation is something like this:
![simple_computer_matmul_naive](/images/simple_computer_matmul_naive.png)

The arithmetic intensity of this implementation on the simple computer is $O(1)$, as each iteration of the innermost loop performs a single multiply-accumulate operation, and only the data for that specific iteration is loaded. This results in $O(N^3)$ data movement and $O(N^3)$ computation, yielding an intensity of $\frac{O(N^3)}{O(N^3)}=O(1)$, which is a factor of $O(N)$ worse than the ideal. This scenario represents the worst-case performance.

### best case
The low intensity of the previous implementation stems from loading single elements from fast memory one at a time, precisely when required. At any given moment, only three matrix elements are held in fast memory. We can boost intensity by making more effective use of fast memory. To demonstrate the best-case scenario, let's suppose that fast memory was large enough to contain $A,B$, and $C$ in their entirety. If this were so, we could allocate space for $C$ in fast memory, transfer all of $A$ and $B$ upfront, execute the three nested loops with all data already present, and then, upon completion, write the entire $C$ matrix back to slow memory at once.
![simple_computer_matmul_best_case](/images/simple_computer_matmul_best_case.png)
In this scenario, since each matrix is moved only once, the data movement is $O(N^2)$. The computation remains the same as before, $O(N^3)$. Examining the ratio of these two, we attain the best-case intensity, $\frac{O(N^3)}{O(N^2)}=O(N)$. This approach, however, is not practical, as the entire problem will typically not fit into fast memory.

### realistic case
We aim to transfer more than three elements at a time between slow and fast memory, but we cannot move the entire matrices at once. A viable compromise is to move sub-tiles of $A$ and $B$ from slow memory to fast memory, making them as large as we can fit. Each pair of input tiles moved into fast memory corresponds to a tile of the output, which can be calculated through a mini-matrix multiplication between the input tiles now resident in fast memory. We then proceed to move the next pair of input tiles to fast memory and repeat the computation.

![simple_computer_matmul_realistic_case](/images/simple_computer_matmul_realistic_case.png)

Here is some pseudocode that corresponds to the diagram above:
```
Allocate A_tile[BN, BN], B_tile[BN,BN], C_tile[BN,BN] in fast memory

# outer loop over tiles of A and B
for i=1...N in steps of size BN:
    for j=1...N in steps of size BN:
        C_tile[: , :] = 0
        for k=1...N in steps of size BN:
            Load A[i : i+BN, k : k+BN] into A_tile
            Load B[k : k+BN, j : j+BN] into B_tile
            
            # inner loop, do a mini matmul between tiles of A and B
            # store the result in C_tile
            for tile_i=1...BN:
                for tile_j=1...BN:
                    for tile_k=1...BN:
                        C_tile[tile_i, tile_j] +=
                            A_tile[tile_i, tile_k] * B_tile[tile_k, tile_j]
            
        # once we have looped over all the tiles along the K dimension of A,B
        # store C_tile back to its place in slow memory
        Store C_tile into C[i : i + BN, j : j+BN]

```
What is the proportion of computation to data movement? How does this measure up against the worst and best cases? We can find answers to these questions by examining the loop structure.

Let's first consider data movement. There are three nested loops on the outside, each iterating from $1$ to $N$ in steps of size $BN$. Each loop runs $\frac{N}{BN}$ times, and with three levels of nesting, the operations inside the nested loop body will execute $(\frac{N}{BN})^3$ times. Within this loop nest, we load two tiles of size $BN^2$, one for each input matrix. Asymptotically, this leads to $O((\frac{N}{BN})^3 * BN^2)$ data movement (we can disregard the storing of `C_tile`, as this occurs only within two of the loop nests and thus happens $\frac{N}{BN}^2$ times). Simplifying this expression gives us $O(\frac{N^3}{BN})$ data movement. Notice that this is a factor of $BN$ less data movement than in the naive approach.

Now let's look at computation. Similar to the above, we have three nested loops, and the inner body of this loop structure will be executed $(\frac{N}{BN})^3$ times. Inside these loop nests, the computation involves a mini-matrix multiplication between two $BN$ by $BN$ tiles; the three nested loops have a total of $O(BN^3)$ steps, which is expected for multiplying two $BN$ by $BN$ matrices. Therefore, the total amount of computation is $O((\frac{N}{BN})^3 * BN^3)$, which simplifies to just $O(N^3)$. This is the expected number of steps for multiplying two $N$ by $N$ matrices, and it matches the naive case.

This tiled method, therefore, has the same number of computation steps as the naive implementation but requires a factor of $O(BN)$ less data movement. The arithmetic intensity calculates to $O(\frac{N^3}{\frac{N^3}{BN}})=O(BN)$. In plain English, this means our achieved arithmetic intensity will increase linearly with the size of the tiles we can fit into fast memory.

### In Summary

The ultimate takeaway is quite intuitive. The highest possible intensity we can reach when multiplying two $N$ by $N$ matrices scales with the matrix dimension $N$. However, attaining this upper bound would necessitate fitting the entire $O(N^2)$-sized problem into fast memory, which is generally not feasible. So, we make a compromise by partitioning the $O(N^2)$-sized problem into numerous smaller $O(BN^2)$-sized problems, and we select $BN$ so that all of our fast memory is fully utilized. The intensity we can then reach scales with $BN$. Consequently, in practice, the achievable intensity is constrained by the amount of fast memory available on our device.

## Parallelized matrix multiplication on a GPU
Considering matrix multiplication on the simple computer helps to build an understanding of how leveraging the memory hierarchy can lead to higher arithmetic intensity, which will be beneficial for maximizing our kernel's performance. However, the simple computer model is somewhat oversimplified; it comprises a two-level memory hierarchy and a compute unit that can operate at a rate of $\tau$ on data in fast memory. Our objective is to write a fast matrix multiplication kernel for a GPU, which brings up the question of how a GPU differs from the simple computer.

At the most basic level, the answer is that GPUs, much like the simple computer, possess a memory hierarchy. But on a GPU, this memory hierarchy is integrated within a hierarchy of concurrent compute units. A diagram of a simple GPU below illustrates this concept.

![simple_gpu](/images/simple_gpu.png)

On this simple GPU, there are three tiers in the combined compute and memory hierarchy.
- At the highest tier is the entire GPU, which possesses a large segment of DRAM (global memory). The GPU is made up of four multiprocessors, each acting as an independent compute unit, running concurrently with the others and all capable of reading from and writing to the same DRAM.
- At the middle tier is a multiprocessor that has its own piece of SRAM (shared memory) and is composed of four cores, which are independent compute units that run concurrently and can all access the same shared memory local to that multiprocessor.
- At the lowest tier is a single compute core that owns some private register memory and is capable of executing a single thread and performing arithmetic independently from the rest of the computer.

### Hierarchical Tiling (simple gpu)
So, how do we utilize this type of computer to carry out a matrix multiplication? The initial useful insight is that the matrix multiplication problem can be decomposed hierarchically into nested tiles. This is fortunate, as a hierarchical algorithm is well-suited for a hierarchical computer.

![matmul_hierarchies](/images/matmul_hierarchies.png)

When we are computing a matrix multiplication $C=A*B$, we can partition the output matrix $C$ into non-overlapping tiles and assign each tile to a distinct compute unit. Each of these output tiles can then be calculated via a matrix multiplication between the corresponding input tiles, independently of the other tiles. Since our machine is hierarchical, it contains compute units within compute units, and similarly, there are matrix multiplications within matrix multiplications. We recursively partition the problem into nested tiles until we arrive at an atomic element of computation, which physically is often a single core of some kind, and logically is a single thread of execution. At this level, the single thread computes a small matrix multiplication between its tiles of the input.
![hierarchy_combined](/images/hierarchy_combined.png)

### Hierarchical Tiling (real gpu)
The diagram above presents a coarse, high-level depiction of what a GPU implementation of hierarchical tiling entails. When putting this into practice with CUDA for an NVIDIA GPU, there are some finer points we need to address. This tiling structure is formed by:
- a series of global, shared, and register memory allocations with fixed dimensions.
- nested loops that manage the positions of the tiles.
- synchronization points among threads running within a single multiprocessor.
- computation at the most granular level, which in this case is a small matrix multiplication executed on the tensor core.

This kernel served as my initial basis, but if you are curious to read about a series of 10 kernels that progressively build up to one like this, I suggest reading [this article](https://siboehm.com/articles/22/CUDA-MMM).

![tiling](/images/my_tiles_2.png)

With this diagram, I am attempting to illustrate the connection between the loop nests and the tiling structure. There are four levels, and each one corresponds to a level in the compute hierarchy, memory hierarchy, and tile shape.

Here is a brief summary of each level from the viewpoint of the relevant compute unit:

*   **CUDA Kernel / GPU level**: The GPU is accessing the three input matrices, $A$, $B$, and $C$, from **global memory**, and writing the resulting matrix $D$ back to global memory. Each thread block iterates over the `K` dimension (also known as the 'inner' dimension) of $A$ and $B$. This loop increments `block_k` in steps of size `BK`. In each iteration, we copy the blue block-tiles from global memory into shared memory.

*   **Thread Block / SM level**: At this stage, the blue sub-tiles of $A$ and $B$ that a specific thread block requires to compute a `BM,BN` tile of the output have been loaded into **shared memory**. This thread block is executing on one of the GPU's 16 SMs, and the shared memory is local to that SM, allowing for fast access. The thread block contains 256 threads, which are organized into 8 warps of 32 threads each. Within the thread block, the `BM,BN` output tile is divided 8 ways, enabling each of the 8 warps to work on the computation concurrently. Each warp loops over the inner dimension within its block tile, incrementing `warp_k` in steps of size `WK`. In every iteration, we transfer the green warp tiles from shared memory to register memory.

*   **Warp / SM Partition**: At this point, the green warp tiles from within the blue block tiles have been moved into **register memory**. It is now the task of a specific warp, running on one of the 4 partitions on the [Turing SM](https://images.app.goo.gl/Z2VVQQgXWTMddBraA), to calculate the `WM` by `WN` output tile. Each warp computes its output tile by performing an outer product between the `WM,WK` tile of A and the `WK,WN` tile of B. Within the three nested loops that execute the outer product, we use an MMA sync operation.

*   **Tensor Core Op**: We finally reach the bottom of the hierarchy, which is a single tensor core operation. This consists of a single hardware-accelerated (16,8) x (8,8) = (16,8) matrix multiply that operates on data in and out of **register memory**.

### Performance considerations on a real GPU
When implementing this structure in a CUDA kernel designed for a specific GPU architecture, there are several factors to consider, given our goal of extracting every last bit of performance from the hardware. I categorize these performance considerations into three main areas, and each optimization discussed in the remainder of this article will fall into one or two of them.

#### Arithmetic intensity as a function of tile dimensions
The need to achieve high arithmetic intensity is the reason for this structure of nested tiles, and the tile dimension is the main parameter we can adjust to control our kernel's arithmetic intensity. In our kernel, we first load data from global memory to shared memory, and then from shared memory into registers. In both transfers, we are loading two rectangular tiles corresponding to the input data from a slower memory level to a faster one, and then ultimately performing a matrix multiplication between these two inputs at the lowest level of the hierarchy. The arithmetic intensity we can expect is a function of our chosen tile dimensions (where larger is better), as detailed below.

![intensity_tile_dims](/images/intensity_tile_dims.png)

- **FLOPs**: In each iteration of the inner loop, every thread block multiplies a $(BM,BK)$-shaped matrix with a $(BK,BN)$-shaped one, resulting in a $(BM,BN)$ output tile. This matrix product involves $2 * BM * BK * BN $ FLOPs (three nested loops over the dimensions, with a multiply and accumulate operation in the innermost loop).
- **memory**: The $(BM,BK)$ and $(BK,BN)$ shaped matrices are read from global memory during each iteration. Since each element is two bytes, this amounts to a total of $2(BM * BK + BK * BN) = 2BK(BM + BN)$ bytes read, and we perform no writes in the inner loop; all writes occur in the kernel epilogue.

Taking the ratio of these two quantities, the arithmetic intensity we should expect for a given block tile size works out neatly to $\frac{BM*BN}{BM+BN} \frac{FLOP}{byte}$. For the thread block level tiles at the second tier of the hierarchy, we will need to select our tile dimensions so that this ratio exceeds the balance point of the tensor cores with respect to global memory, though we will be constrained by the amount of shared memory. Similarly, for the warp tiles at the next level down in the hierarchy, we will aim to choose tile dimensions such that this ratio is greater than the tensor cores' balance point with respect to shared memory, but here we will be limited by the size of register memory. The former task proves to be slightly more challenging than the latter.

#### Overlap between compute and data movement
The roofline model provides an upper limit on arithmetic throughput, given by $T_{max}=min(\beta * I, \tau)$. To reach this upper bound, we must achieve perfect overlap between computation and data movement. To understand why, consider a scenario where we achieve an arithmetic intensity high enough to place us in the compute-bound region of the roofline model. At this juncture, for our achieved throughput to actually match the upper bound $T_{max}=\tau$, our computation must be continuous; any time our compute units are idle will mean our achieved throughput falls short of the machine's peak, $\tau$. There are several reasons our compute units might be idle, including memory latency, data dependencies, and synchronization points.
![compute_data_movement_overlap](/images/compute_data_movement_overlap.png)
As depicted above, our initial loop structure exhibits some inefficiencies in this area.

#### Maximizing memory bandwidth
According to [unofficial benchmarks](https://arxiv.org/pdf/1903.07486), the maximum achievable global memory bandwidth on the T4 is approximately 220 GB/sec, and the best shared memory bandwidth is around 3662 GB/sec. However, a non-optimized kernel will only attain a fraction of these figures. The primary consideration is the access pattern; when groups of adjacent threads request memory, some mappings of threads to data in memory are more efficient than others. The hardware implementing global memory and shared memory functions differently, so an access pattern that is optimal for reading from shared memory may not be optimal for reading from global memory.

The main factor for global memory access is known as coalescing; in short, maximum global memory bandwidth is achieved when adjacent threads access adjacent data in global memory (as explained [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)). Shared memory is explored in a [subsequent](#background-bank-conflicts-and-wavefronts) chapter.

### How to use Tensor Cores
This section provides a concise overview of the mechanics involved in using tensor cores.

All tensor core operations are executed at the warp level within the compute hierarchy; 32 threads work together to load data into their registers and then synchronously perform a small, hardware-accelerated matrix multiplication. When designing tensor core algorithms, we should consider the warp as an atomic unit of computation, even though a warp actually consists of 32 threads, each capable of independent work. In contrast, if we were writing a GEMM kernel without tensor cores, individual threads performing scalar multiply-accumulate operations would be our atomic compute element.

Tensor cores can be accessed through two different methods. The first is via the `wmma` [API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#wmma-description), which is included in the CUDA toolkit. `wmma` is generally seen as the more portable but less performant method for programming tensor cores. I abandoned it rather quickly, as it abstracts away the process of loading input data from shared memory into register memory, and it turns out that certain details in this process are crucial for performance.

The alternative method is to use the `mma` family of instructions, which are part of PTX; this choice offers more flexibility and better performance than the `wmma` approach. PTX serves as an intermediate representation for NVIDIA GPUs, positioned at a lower level than CUDA but higher than SASS (the assembly language that NVIDIA GPUs execute). PTX can be inlined within a kernel to invoke tensor cores.

The specific PTX instruction I utilized is `mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16` (documentation available [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-mma-m16n8k8)), and each component of this instruction has a specific meaning:
* `mma`: signifies that we are performing a matrix multiply-accumulate operation.
* `sync`: indicates that this instruction is synchronous, meaning all 32 threads will pause until all 32 have completed before continuing execution.
* `aligned`: requires that all 32 threads in a warp execute this instruction; if fewer than 32 threads in a warp attempt to execute it, the behavior is not defined.
* `m16n8k8`: this is the identifier for the matrix fragment shape. It specifies that the fragment of matrix
$A$ has a shape of (16,8), the fragment of $B$ has a shape of (8,8), and the fragments of $D$ and $C$ have a shape of (8,8). (Recall that the GEMM formula is $D = \alpha * A * B + \beta * C$). If you examine the linked PTX documentation, you will find many different shapes available; however, the Turing/Volta architectures only support a limited subset. Ampere supports more shapes, and Hopper supports even more.
* `row`: specifies that the $A$ fragment should be stored in registers using a row-major layout.
* `col`: specifies that the $B$ fragment should be stored in registers using a column-major layout.
* `f16`: indicates that $D$ is an fp16 matrix.
* `f16`: indicates that $A$ is an fp16 matrix.
* `f16`: indicates that $B$ is an fp16 matrix.
* `f16`: indicates that $C$ is an fp16 matrix.

Each `mma.sync` instruction requires a particular layout of fragment elements distributed across the registers of the 32 threads in a warp; these layouts are detailed in the PTX documentation. Here is the layout for `m16n8k8`:
![matrix_fragments](/images/mma_fragments.png)

These diagrams illustrate a mapping between threads, registers, and matrix elements:
* `T0, T1, T2 ...` represents the index of a thread. Thread indices in these diagrams span from 0 to 31, as there are 32 threads in a warp.
* `a0, a1, a2, ... b0, b1, b2, ... c0, c1, c2` refer to the registers that contain the matrix elements.
* The location of each thread/register pair indicates which matrix elements are placed in which registers of which thread. For instance, `T0: {a0,a1}` is situated at the top-left corner of matrix fragment A, which means elements `(0,0)` and `(0,1)` of this fragment are stored in registers `a0` and `a1` of thread 0.

Fortunately, another PTX instruction called `ldmatrix` (documentation [here](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix)) exists to load a rectangular tile of data from shared memory and shuffle the matrix elements within a warp to create this specific layout for us. It can also optionally transpose matrix elements as they are moved from shared memory to registers, which is convenient for matrix B above, as it is in a column-major, or "transposed," layout.

The inner loop of our kernels will involve repeatedly calling `ldmatrix` to move data from shared memory into register memory, and then repeatedly invoking the `m16n8k8` variant of `mma.sync` to multiply these tiles together using the tensor core. For this project, I worked with a Turing architecture GPU; on Ampere, the tensor core API is very similar but supports more matrix shapes. On Hopper, the API is significantly expanded, with new PTX instructions that permit a group of 128 threads to asynchronously execute a much larger matrix multiplication than `m16n8k8`.

# Kernels

For the remainder of this piece, I will go over a series of kernels that brought me to approximately 96% of cuBLAS-level performance on a tensor core GEMM for 8192x8192 matrices. Each kernel improves upon the previous one, and the central themes are:
1. [hierarchical tiling](#kernel-1---hierarchical-tiling)
2. [vectorized/unrolled gmem->smem transfer](#kernel-2---vectorized-memory-copy-and-loop-unrolling)
3. [shared memory swizzling](#swizzling)
4. [makeshift async copy](#kernel-4---makeshift-async-copy)
5. [tune tile dimensions](#tune-tile-dimensions)
6. [optimized index calculation](#kernel-5---optimize-index-calculation)
7. [double buffering](#kernel-6---double-buffering)

## Kernel 1 - Hierarchical Tiling
The first kernel I developed is an implementation of the hierarchical tiling structure depicted [earlier](#hierarchical-tiling-real-gpu). Here is pseudocode for the loop structure that carries out the matrix multiplication.

```c++
// outer loop over block tiles
for (block_k = 0; block_k < K; block_k += BK)
{
    // global memory to shared memory transfer
    A_smem[:,:] = A_gmem[block_m:block_m+BM, block_k:block_k+BK]
    B_smem[:,:] = B_gmem[block_k:block_k+BK, block_n:block_n+BN]
    
    // synchronize across the thread block
    __syncthreads();

    for (warp_k = 0; warp_k < BK; warp_k += WK)
    {
        A_reg[: ,:] = A_smem[warp_m:warp_m+WM, warp_k:warp_k+WK]
        B_reg[:, :] = B_smem[warp_k:warp_k+WK, warp_n:warp_n+WN]

        for (mma_k = 0; mma_k < WK; mma_k += MMA_K)
        {
            for (mma_m = 0; mma_m < WM; mma_m += MMA_M)
            {
                for (mma_n = 0; mma_n < WN; mma_n += MMA_N)
                {
                    mma_sync_m16n8k8(
                        acc_reg[mma_m:mma_m+MMA_M, mma_n:mma_n+MMA_N],
                        A_reg[mma_m:mma_m+MMA_M, mma_k:mma_k+MMA_K],
                        B_reg[mma_k:mma_k+MMA_K, mma_n:mma_n+MMA_N],
                        acc_reg[mma_m:mma_m+MMA_M, mma_n:mma_n+MMA_N]
                    )

                }
            }
        }
    }
    __syncthreads();

}
```
The 8% of cuBLAS throughput it attains serves as our starting point. The rest of this article explores some techniques I employed to make it run faster.

![table1](/images/table1.png)


## Kernel 2 - Vectorized memory copy and loop unrolling
To enhance the performance of our code, we must first understand why it is slow. When developing CUDA kernels, the premier tool for this analysis is NSight Compute, a profiler created by NVIDIA that provides a wealth of detailed metrics on how a kernel interacts with the hardware. The first section I usually examine is called "Warp State Statistics." As a kernel runs, each warp receives instructions from a scheduler. In a perfect world, the scheduler would be able to issue a new instruction every single clock cycle. In reality, creating a kernel that can issue an instruction every cycle is extremely difficult, as there are numerous reasons why a warp might be unable to execute its next instruction on a given cycle and will instead "stall," or do nothing. The causes for stalling can range from capacity limitations of various hardware pipelines and memory latency to synchronization points in our kernel that force all threads on an SM to wait for the other threads to catch up. The Warp State Statistics section reveals how many clock cycles the average warp spends stalled for each instruction issued, broken down into many different categories. This provides the necessary information to focus our optimization efforts on the least efficient parts of our kernel. Below is a screenshot of the Warp State section for Kernel 1.
![warp_state_kernel1](/images/warp_state_kernel1.png)
The "Warp Cycles Per Issued Instruction" field indicates that, on average, for each instruction issued, warps spend approximately 30 cycles idle, and the table below reveals that 16 of these 30 cycles are attributed to the "Long Scoreboard" stall category.

[Scoreboarding](https://en.wikipedia.org/wiki/Scoreboarding) is a hardware-implemented algorithm in most processors for monitoring when the data dependencies for the next instruction have arrived in the necessary registers for execution. Most modern CPUs can reorder instructions on-the-fly, allowing instructions with ready operands to execute before those whose operands have not yet arrived in registers. This reordering is managed by the hardware, subject to constraints imposed by data dependencies between consecutive instructions. This technique is known as [out-of-order execution](https://en.wikipedia.org/wiki/Out-of-order_execution) and is a rather sophisticated method for hiding latency. GPUs do not reorder instructions during execution, likely because the required logic would consume a significant amount of precious on-chip transistors, and since GPUs are designed for [throughput](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-benefits-of-using-gpus), these transistors are better allocated to components like tensor cores.

GPUs do, however, track data dependencies, but they rely much more on the compiler for assistance compared to CPUs. When the data required to execute the next instruction has not yet arrived in register memory, the executing warp simply waits for its data to arrive. The "Long Scoreboard Stall" metric approximates the average number of cycles that warps spend stalled while waiting for data dependencies. The fact that this stall reason constitutes about 50% of all idle warp cycles indicates that the performance of Kernel 1 is predominantly limited by memory latency. This suggests we should concentrate on the code responsible for moving data from global memory onto the chip and find ways to minimize the latency per byte transferred.

Reading a rectangular tile of data from global memory and writing it to shared memory is the first action that happens in each iteration of the kernel's outer loop. The most straightforward way to accomplish this is for adjacent threads to access adjacent values in global memory and then write the data to shared memory in the same layout it had in global memory. This access pattern is optimal for both reading from global memory and writing to shared memory. Here is the initial data transfer function I wrote:

```c++
__device__ void tileMemcpy(
    half* src,
    half* dst,
    const unsigned int src_stride,
    const unsigned int tile_rows,
    const unsigned int tile_cols
)
{
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int num_threads = blockDim.x * blockDim.y;
    
    // # of threads is multiple of # of columns in the tile
    assert(num_threads % tile_cols == 0);
    
    // assign each thread a row/column in the tile, calculate the row step
    const unsigned int row_step = num_threads / tile_cols;
    const unsigned int thread_row = thread_idx / tile_cols;
    const unsigned int thread_col = thread_idx % tile_cols;
    
    for (unsigned int r = thread_row; r < tile_rows; r+=row_step)
    {
        dst[r * tile_cols + thread_col] =  src[r * src_stride + thread_col];
    }
}
```
When we examine the SASS corresponding to this `tileMemcpy` function in [godbolt](https://godbolt.org/z/1MeavE3GG), we can see that the copy operation `dst[...] = src[...]` inside the loop compiles into two operations from the lower-level perspective of SASS: a two-byte load from global memory (`LDG.U16` in SASS), followed by a two-byte store (`STS.U16`), in addition to a series of index calculations and loop overhead. The long scoreboard stall delays the store operation until the value being loaded has arrived in the register.

Here is a depiction of how this loop executes, from the perspective of a single thread:
![memory_latency](/images/memory_latency.png)
Latency between the load and the store is unavoidable: a request is dispatched to a DRAM controller, data is retrieved from DRAM, and then sent over a bus. We cannot eliminate this latency unless we can alter the laws of physics or invent a time machine. What we can do, however, is conceal it.

Latency hiding is a fundamental concept in computing, and at its heart, it is quite simple. It merely means that if we are performing an operation $X$ that incurs some latency, we should be engaged in other useful work while $X$ is in progress, rather than waiting idly. For example, if I wake up wanting an omelette, I would first turn on the stove to let the pan heat up, and while that is happening, I would crack the eggs and grate the cheese. This sequence of actions hides the latency of heating the pan with the tasks of cracking eggs and grating cheese. If I am hungry and anxious to eat the finished omelette as soon as possible, it would be foolish to stand by and watch the pan warm up.

The same principle is applicable to hiding the latency of global memory loads in `tileMemcpy`. Since the copy operation is performed within a loop, each thread executes multiple loads and stores in a sequence like `load (stall) store, load (stall) store, ...`. What if we could reorganize these operations so that the sequence becomes `load load load (stall) store, store, store`? In this latter arrangement, the data requested by the three loads would be in transit simultaneously, and we can say that the latency of each load is being hidden by the other loads. The most straightforward way to achieve this latter ordering is by unrolling the loop in `tileMemcpy`. If we can unroll the loop, `nvcc` should be intelligent enough to reorder the instructions so that the global memory loads conceal each other's latency. In this scenario, the compiler is performing for us what a CPU would do in hardware on the fly.

If we wish to unroll the loop, the number of iterations must be known at compile time. The number of loop iterations depends on the number of threads per block and the block tile dimensions. Since both of these are fixed at compile time, we can pass them as template parameters to `tileMemcpy`, calculate the number of iterations as a function of these parameters, and add a `#pragma unroll` directive to achieve our goal.

```c++
template<unsigned int TILE_ROWS,
unsigned int TILE_COLS,
unsigned int NUM_THREADS>
__device__ __forceinline__ void tileMemcpyUnrolled(
    half* src,
    half* dst,
    const unsigned int src_stride
)
{
    // # of threads is multiple of # of columns in the tile
    static_assert(NUM_THREADS % TILE_COLS == 0);
    
    // flatten out 2d grid of threads into in order of increasing threadIdx.x
    const unsigned int thread_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // assign each thread a row/column in the tile, calculate how many iterations we need
    // to cover the whole tile
    constexpr unsigned int ROW_STEP = NUM_THREADS / TILE_COLS;
    constexpr unsigned int NUM_ITERS = TILE_ROWS / ROW_STEP;
    unsigned int thread_row = thread_idx / TILE_COLS;
    const unsigned int thread_col = thread_idx % TILE_COLS;
    
    #pragma unroll
    for (unsigned int i = 0; i < NUM_ITERS; i++)
    {
        dst[thread_row * TILE_COLS + thread_col] =  src[thread_row * src_stride + thread_col];
        thread_row += ROW_STEP;
    }
    
}
```
This change gives us something more like the following:
![memory_latency_unrolled](/images/memory_latency_unrolled.png)
In the original version, the total latency of the copy operation is roughly proportional to the device's memory latency multiplied by the number of loop iterations. After unrolling the loop, the total latency should be reduced from the initial version by a factor related to the number of loads the compiler chooses to overlap (approximately).

The other relatively simple optimization we can implement here is to increase the number of bytes loaded per instruction. Our current load operation compiles to `LDG.U16`, where each instruction fetches 16 bits/2 bytes from DRAM. The widest load instruction in SASS is `LDG.128`, which loads 128 bits/16 bytes. Since our kernel is limited by memory latency rather than memory bandwidth, using a wider load instruction will allow us to experience the same latency per memory request but transfer more bytes with each request. We are effectively amortizing the latency over a larger number of bytes moved, which represents a gain in efficiency.

![memory_latency_vectorized](/images/memory_latency_vectorized.png)

A fast and somewhat crude way to achieve this is by `reinterpret_cast`ing the `src` and `dst` pointers from `half` to `float4`, and then adjusting the index and loop calculations accordingly. Here is a [godbolt link](https://godbolt.org/z/v3T3x14ns) to a kernel with the vectorized and unrolled memory copy, and the code can be found [here](https://github.com/alexarmbr/matmul-playground/blob/main/src/device_utils.cuh#L73).

These optimizations to the `memcpy` function boost the throughput by about 3x over the first kernel. However, there is still a considerable distance to cover before we can match cuBLAS-level performance.
![table2](/images/table2.png)

## Kernel 3 - Shared Memory Swizzling
Let's return to the warp state section of NSight Compute.
![kernel2_nsight_compute](/images/kernel2_nsight_compute.png)
The long scoreboard stall is no longer the primary cause of warp stalls, and our kernel's performance improved by approximately 3x after implementing the optimizations from the last section. Warps are now spending an average of about 19 cycles stalled per issued instruction due to a phenomenon called "MIO Throttling." What exactly is MIO Throttling, and how can we mitigate it? According to the NSight Compute [documentation](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html), this signifies:
>Warp was stalled waiting for the MIO (memory input/output) instruction queue to be not full. This stall reason is high in cases of extreme utilization of the MIO pipelines, which include special math instructions, dynamic branches, as well as shared memory instructions.

In our situation, this stalling is almost certainly caused by shared memory instructions, as our kernel contains very few dynamic branches and no trigonometry or other [special math](https://developer.nvidia.com/cuda-math-library) functions. To be precise, it is a result of shared memory bank conflicts. According to a post [here](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731/2?u=a14armbr), two indicators of shared memory bank conflicts are an extremely high L1/TEX throughput number (currently at 97% of peak) and MIO Throttle stalls; both are secondary consequences of shared memory bank conflicts. I learned at this stage that if you have a kernel whose performance is being severely impacted by shared memory bank conflicts, it is not immediately obvious from a glance at NSight Compute, but the necessary information is definitely present. I found that to identify where shared memory bank conflicts were happening and to grasp their severity, I needed to learn the terminology of a "wavefront." To understand this term, some background on shared memory is necessary.

### Background: Bank Conflicts and Wavefronts
From a CUDA program's point of view, shared memory operates as described below (the official guide is [here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory-5-x)). When you declare a `__shared__` array in your kernel, it maps to physical memory situated on a specific streaming multiprocessor. As a result, this array is quick to access but is only available to threads on that SM, which, in CUDA terminology, means that shared memory arrays are local to a particular thread block. Physically, the memory is distributed across 32 "banks," with each bank storing an adjacent 4 bytes, as illustrated here:
![shmem_1](/images/shmem_1.png)
Each bank is capable of providing a single 4-byte value per clock cycle. If our objective is to maximize our read and write bandwidth from shared memory, we must bear this in mind when choosing an access pattern. Full bandwidth is reached when the 32 threads in a warp distribute their accesses uniformly across the 32 banks. Bank "conflicts" happen when a single bank must supply data to more than one thread for a single request. To demonstrate how the concepts of bank conflicts and wavefronts are linked, here are three scenarios, all set in a simplified world with 4 threads and 4 memory banks.
![bank_conflicts](/images/bank_conflicts_wavefronts.png)
When loading from or storing to shared memory, each thread requests a specific memory address that, in our simplified model, falls into one of the four memory banks. In the first scenario, each thread accesses data in a different bank, and the hardware determines that these four accesses can be consolidated into a single transaction for processing; this transaction is termed a [wavefront](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#id26). In the second scenario, the four threads access addresses that fall into just two of the four banks. Since each bank can only send one word at a time, the hardware groups these four requests into two separate wavefronts, and the memory hardware processes these two wavefronts sequentially. The third scenario represents the worst case, where the four threads access addresses that all map to the 0th memory bank; in this situation, four separate wavefronts are needed to service the transactions from the four threads.

For four threads accessing four bytes, the "ideal" number of wavefronts is one, because (ideally) no matter which threads are accessing which bytes, we should be able to organize our data so that all our accesses are spread evenly across the banks. For example, scenario three as depicted is suboptimal, but we could make it ideal by transposing the bytes in shared memory, which would result in the four accesses being distributed evenly across the four banks. However, for the layout as shown, the actual number of wavefronts is four.

NSight Compute provides, for each memory access:
1. the ideal number of wavefronts.
2. the actual number of wavefronts.
3. the number of excessive wavefronts, which is simply the result of 2 minus 1.

Based on the analysis above, if our code exhibits an $n$-way bank conflict, then $n$ should be equal to $\frac{actual\ wavefronts}{ideal\ wavefronts}$. We want the actual number to match the ideal, which often requires careful consideration of how data is laid out and how threads are accessing it.

### ldmatrix bank conflicts
Here is a screenshot showing the per-instruction actual versus ideal wavefronts in NSight Compute:
![l1_wavefronts_source_view](/images/l1_wavefronts_source_view.png)
These `ldmatrix` instructions are transferring data from shared memory into thread-local register memory in preparation for the MMA operations. NSight Compute indicates that the ratio of actual to ideal wavefronts is approximately 8, which points to this memory access causing an 8-way bank conflict. To devise a strategy for resolving this performance bottleneck, we must first understand its cause.

In the tiling structure presented for Kernel 1, during each iteration of the warp loop (the green one), a single warp is tasked with reading a 64x64 tile of data from shared memory and writing it to registers. The bank conflicts arise during these shared memory reads. In the visualization below, the top portion shows a highly zoomed-out view of one of these 64x64 tiles, with the layout across memory banks indicated by the color of the columns. We can observe that a row of 64 elements, each 2 bytes, neatly spans the 32 memory banks.
The bottom portion provides a zoomed-in view of a single 8x8 tile that is moved from shared memory into registers by `ldmatrix`. Each warp iterates over its own local 64x64 tile in 8x8 increments, invoking `ldmatrix` on each small tile; this PTX instruction loads values from shared memory and then shuffles the loaded data among the registers in a warp to match the register layout expected by the tensor core instruction.
![mma_tile_zoom_in](/images/mma_tile_zoom_in.png)
The internal operations of `ldmatrix` are somewhat opaque; it compiles to a single SASS instruction, `LDSM...`, rather than multiple explicit shared memory loads and register shuffles, as one might anticipate. However, we do not need to understand the inner workings of `ldmatrix` to see why the 8-way bank conflict occurs each time we call it. The 8-way bank conflict is, in fact, an unavoidable consequence of each row in a given tile being spread across the same four memory banks. One wavefront is needed to read each row, and since there are eight rows, this results in eight wavefronts. Ideally, if the eight rows in each tile were distributed evenly across the thirty-two memory banks, the entire tile could be read with just a single wavefront. Reading these tiles is part of the kernel's inner loop; for $8192$x$8192$ operands, we read a total of $ (8192/8)^3=1,073,741,824$ of these tiles, which amounts to a massive number of bank conflicts. Therefore, if we are concerned about performance, it is well worth the time to resolve this issue.

### Padding
To create a bank-conflict-free kernel, we must rearrange the data layout in shared memory so that we can read from and write to it without any excessive wavefronts. The difficulty arises from the fact that the thread-to-data mapping for shared memory reads is different from that for shared memory writes. When writing, adjacent threads write to adjacent values in a row, whereas when reading, adjacent threads read adjacent values down a column.

![row_vs_column_shmem_access](/images/row_vs_column_shmem_access.png)

This is a frequent issue in kernels that utilize 2D shared memory tiles, and the conventional solution is to add a small amount of padding (i.e., empty space) to the end of each row in the shared memory array. If we introduce this padding in such a way that a single row of our array no longer aligns perfectly with the 32 memory banks, then adjacent values in a column will no longer fall into the same bank, which allows us to read columns without any excessive wavefronts. This concept is better understood through a picture than through words; here again is a simplified case of a mini-array (4 columns and 4 rows) stored on a mini-GPU with only 4 memory banks:
![simple_smem_padding](/images/simple_smem_padding.png)
Array elements are color-coded by their column. Note that in the no-padding case, all array elements within a given column fall into the same memory bank. After adding the column of padding, the array elements in any given column are distributed across all 4 memory banks. The padding technique could be employed here to completely eliminate bank conflicts. Since we are using [vectorized](#kernel-2---vectorized-memory-copy-and-loop-unrolling) writes to shared memory, we are writing in 16-byte chunks at a time, and each chunk must be [aligned](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses). Adding 16 bytes of padding to each row of shared memory would cause each 8x8 MMA tile to be spread across all 32 memory banks (the exercise of convincing yourself of this is left to the reader).

The downside of using the padding technique is that it forces us to allocate extra, unused space in shared memory. In Kernel 2, the shared memory tile for $A$ is 256x64, and the tile for $B$ is 128x64. If we were to add an extra 16-byte, or 8-element, column to both of these, it would increase our shared memory allocation by 25%, for a total increase of 6144 bytes. This wasted space becomes a significant drawback, as shared memory is a very precious resource when writing a high-performance kernel—this becomes particularly evident later on when using a technique called double buffering, where each threadblock in future kernels will end up using 100% of the 65,536 bytes of shared memory on each SM. So, we should ask whether there is a way to eliminate bank conflicts without wasting any shared memory. It turns out this is entirely possible!

### Swizzling (toy example)
Swizzling is likely my favorite technique that I discovered while working on this project. The term "swizzle" has several different meanings; in the context of cocktails, it refers to [stirring](https://en.wikipedia.org/wiki/Swizzle_stick), while in the context of GPUs, it means to [rearrange](https://en.wikipedia.org/wiki/Swizzling_(computer_graphics)). In our context of eliminating shared memory bank conflicts in 2D tiles of data, swizzling involves permuting the elements within a tile of shared memory so that we can access the data without any bank conflicts. This is one of those techniques that appeared to be black magic to me until I took the time to understand it, and now I can appreciate its cleverness and elegance.

In our 4x4 tile example, we add padding because it shifts the alignment between data and memory banks in a favorable manner. Swizzling is founded on the insight that we do not need the extra padding bytes to distribute column elements evenly across the memory banks. Instead, we can simply determine a permutation of matrix elements that spreads the columns in the correct way and apply this permutation when we write to shared memory. Here is a demonstration of a "swizzle"—that is, a permutation of elements that can eliminate bank conflicts.
![simple_smem_swizzled](/images/simple_smem_swizzled.png)
It is important to remember at this point that our shared memory layout must meet two criteria: bank-conflict-free row access for writing, and bank-conflict-free column access for reading.

In all three scenarios, each row is laid out consecutively in memory and distributed across all four memory banks, which means each row can be written without any bank conflicts. The key observation here is that when we apply our permutation, or "swizzle," we do not want to permute elements across different rows, only within rows; otherwise, we might lose the property of bank-conflict-free writes.

The issue that led us to consider shared memory layouts was the bank conflicts that arise when we read columns. Adding padding resolves these bank conflicts, but at the cost of wasted shared memory. Swizzling offers the best of both worlds: we can read columns without bank conflicts, and no shared memory is squandered. So, how do we go about applying this permutation?

The swizzle shown previously can be implemented as a function `f` that maps original indices to new indices. If `A` represents the original array, `A_s` the swizzled array, and `i` the index of an element, then the relationship is `A_s[f(i)] = A[i]`. So, what is this function `f`?

Since `f` operates on array indices, we should consider the different ways these indices can be represented and interpreted:
![simple_smem_indices](/images/simple_smem_indices.png)
On the far left are the 2D row and column indices. Moving to the center, these indices can be linearized into a sequential (and in this case, row-major) ordering of the 16 elements in the array. Moving to the right, when we examine the sequential indices in their binary form, we can see that the 2D structure is reflected in the index bits. The two least significant bits of the index encode the column, while the other two bits encode the row. As a spoiler, `f` will operate from the perspective of the view on the right, which is the binary representation of the flattened array index. Here are two observations about what `f` must accomplish:
*   To prevent bank conflicts on write, we aim to permute elements only within a row, meaning no elements should change their row. This implies that `f` should alter the bits that encode the column and leave the bits that encode the row untouched.
*   We want to apply a different permutation to each row, and for any given column, we want the elements of that column to be distributed across all four columns in the swizzled array.

We can achieve both of these objectives by using the XOR function, specifically by XORing the row bits of each element with its column bits and using the result as the new column bits. Here is a row-by-row analysis that demonstrates how XORing the column bits with the row bits rearranges values within a row:
![swizzled_rows](/images/swizzled_rows.png)
The function `f` that accomplishes this for us is `f(i) = i ^ ((i & 0b1100) >> 2)`. The mask `0b1100` selects the two column bits from `i`; these two bits are then shifted right by two positions to align with the two row bits of `i`, and then we perform the XOR operation. The column bits of `i` remain unchanged.

Here is a visualization of the outcome of applying this function to all rows collectively:
![2d-swizzle](/images/2d-swizzle.png)

### Swizzling (real world)
Now, we must determine how to apply this technique to permute our shared memory layout so that we can read a single 8x8 MMA tile with zero excessive wavefronts. As a refresher, here is a view of our current shared memory layout, with a single tile of interest highlighted.
![mma_tile_zoom_in_blank](/images/mma_tile_zoom_in_blank.png)

Our objective is to devise a swizzle function that distributes the 8 rows of this tile across all 32 memory banks, instead of having all 8 rows confined to just 4 memory banks, which is the current situation. From the perspective of the full tile, the rows of the tile above would be spread out like this.

![mma_tile_zoom_in_swizzle](/images/mma_tile_zoom_in_swizzle.png)

To determine the appropriate swizzle function to use, let's examine the binary representation of an index into this tile and assign it a structure that aligns with our tiling scheme.

![swizzle_index_groups](/images/swizzle_index_groups.png)

Here are some notes on what our swizzling function should and should not do:
*   We want to maintain the eight elements in each MMA tile row as a contiguous block. In other words, eight adjacent elements within a single row of an 8x8 MMA tile will remain together after we apply the swizzle. This means our swizzle function will not modify the orange bits.
*   Bank conflicts arise because the 8 rows within an MMA tile are all perfectly aligned on top of one another. Within an MMA tile, we want to distribute these 8 rows horizontally across the entire warp tile. The blue bits encode the position of each MMA tile within the 64-element-wide warp tile, so these are the bits we want our swizzle function to alter.
*   We do not want to move elements between rows, so our swizzle function will not change the green row bits. However, these green row bits provide a useful alternating pattern that we can XOR with the blue bits to shuffle the MMA tiles within their row.
*   Again, we do not want to move elements between rows, and the black bits (the most significant ones shown in this diagram) encode the starting row of each MMA tile. Our swizzle function will ignore them.

So, all of this implies that for each index, we want to take the blue bits, XOR them with the green bits, and then replace the original blue bits with the result of this XOR operation. If `i` is the index we wish to swizzle, this can be expressed as:
![swizzled_vs_unswizzled](/images/swizzled_vs_unswizzled.png)
And just like that, we have eliminated the bank conflicts. Swizzling requires a bit more effort to figure out than the padding technique; the choice of swizzle function is dependent on the shared memory array dimensions and the vector width used for reads/writes (e.g., `float4`, `float2`, `int`, etc.). As a result, using swizzling adds an extra consideration each time we contemplate changing either of these parameters. However, if you aim to eliminate bank conflicts without increasing your shared memory footprint, swizzling becomes essential. I find it to be a very elegant and clever solution; if you compare kernel 2 with kernel 3, only about four lines of code are changed, and these four lines are the addition of the swizzle into the shared memory index calculation.

I figured all this out by studying the `Swizzle` class implemented [here](https://github.com/NVIDIA/cutlass/blob/main/python/pycute/swizzle.py) in the CUTLASS repository. Through its three parameters, `bits`, `base`, and `shift`, this class represents a family of swizzle functions that shift and XOR bits of array indices. I have also encountered examples of more exotic swizzle functions (see slide 27 [here](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)) that extend beyond what can be represented by the CUTLASS implementation. I found it helpful to visualize the permutations applied by different swizzle functions; to aid with this, I wrote a small Python [script](https://github.com/alexarmbr/matmul-playground/blob/main/scripts/shmem_layout_viz.py) that pretty-prints arrays, applies swizzle functions, and counts bank conflicts.

Eliminating bank conflicts yields an approximate 2x speedup and brings us to about 50% of the cuBLAS-level throughput.
![table3](/images/table3.png)

## Kernel 4 - Makeshift Async Copy
Each optimization targets the least efficient part of the preceding kernel. After applying an optimization, if it is successful, the kernel's bottleneck should shift. Before we fixed the shared memory bank conflicts, the shared memory operations in the inner loop were the primary bottleneck. After eliminating those bank conflicts, the inner loop became much more efficient, and the bottleneck has once again become the latency of the global memory to shared memory transfer. This was previously addressed with vectorization and loop unrolling in [Kernel 2](#kernel-2---vectorized-memory-copy-and-loop-unrolling), but after fixing the bank conflicts, NSight Compute now indicates that there is still more latency to hide. Here is pseudocode of the current loop nests, with a focused view on the code that requires improvement:
![long_scoreboard_stall_kernel3](/images/long_scoreboard_stall_kernel3.png)
Once more, the problem is that the line performing the global memory to shared memory copy:

```c++
dst_float4[dst_index] = src_float4[src_index];
// shared memory        // global memory
```

This^ operation is blocking from the hardware's perspective, meaning that when a given thread executes the resulting assembly, it will be stalled for the entire duration it takes for data to arrive from global memory. The line above is equivalent to this:

```c++
float4 tmp = src_float4[src_index]; // global memory to register
dst_float4[dst_index] =  tmp; // register to shared memory
```
The global-memory-to-register transfer, which is the first line, incurs latency because the data is coming from off the chip. When it is time to store from the register to shared memory (the second line), the hardware detects that the necessary data from global memory has not yet arrived in `tmp`, and execution stalls until it does. In [Kernel 2](#kernel-2---vectorized-memory-copy-and-loop-unrolling), we addressed this performance issue by amortizing the latency over more data per transaction (vectorizing) and by helping the compiler to interleave multiple loads and stores to hide latency (loop unrolling). Yet, NSight Compute shows that even after these optimizations, this type of stall, on this specific line, is responsible for about 20% of the total clock cycles the kernel spends stalled.

The key insight here is that if we decompose the `dst[...] = src[...]` line into its two fundamental parts, we can separate them so that other useful work can be performed while the data is in transit from global memory.
The general strategy is to prefetch data from global memory into register storage one `block_k` step ahead of the `block_k` we are currently processing. At a high level, we want to transition from this:
```c++
float4 tmp = src_float4[src_index]; // global memory to register
// (stall while we wait for data to arrive from memory)
dst_float4[dst_index] =  tmp; // register to shared memory
{
    // compute inner loop for current block tile
}
```

to this:
```c++
float4 tmp = src_float4[src_index]; // global memory to register
{
    // compute inner loop for previous block tile
}
dst_float4[dst_index] =  tmp; // register to shared memory
```

The crucial improvement being made here is that we are initiating the data load from global memory corresponding to `block_k` while concurrently performing the computation for `block_k-1`. In doing so, we are hiding the latency of loading the `block_k` tiles of $A$ and $B$ with the computation associated with the `block_k-1` tiles.

![concurrent_fetch_compute](/images/concurrent_fetch_compute.png)

This enhanced overlapping of data movement and computation is achieved by:
- adding new register storage to hold the data that is prefetched from global memory.
- breaking up the global-to-shared memory transfer into its two components and placing these two components on opposite sides of the inner loop (which iterates over warp tiles and MMA tiles).
- and adjusting the positions of the two `__syncthreads()` calls in the outer loop to enable the desired concurrency while still preventing race conditions.

Here is a before-and-after pseudocode representation showing how the data movement is altered.
![prefetch](/images/prefetch.png)

This change yields a significant speedup over the previous kernel, bringing us to approximately 70% of the HGEMM kernel's performance.

![table4](/images/table4.png)

### GPU occupancy (digression)
The potential downside of this optimization is that it demands additional register storage; each thread block now stores two extra block tiles' worth of data in register memory. According to the Launch Statistics section in NSight Compute, we have increased from using 104 registers per thread in Kernel 3 to 166 registers per thread in Kernel 4. This increased resource usage per thread has the *potential* to negatively impact kernel performance because it can affect how many threads the hardware can execute concurrently. This is a brief digression on why increasing register use per thread could potentially harm performance, but why, in this specific case, it does not.

This brings us to a concept called occupancy, which is fundamental to the CUDA hardware and software model. Each streaming multiprocessor (SM) maintains the execution state for blocks, warps, and threads (including shared memory, registers, and program counters) on-chip for as many thread blocks as it can accommodate. The number of thread blocks that can fit on an SM is determined by:
1. the amount of shared memory, registers per thread, and number of threads that each thread block requires for execution (this is a characteristic of a given kernel and its launch configuration).
2. the total amount of shared memory, registers per thread, and number of threads that the SM can manage simultaneously (this is a property of the device and improves with each generation).

If a particular kernel implementation and launch configuration require only a small number of registers, a few threads, and a minimal amount of shared memory, an SM can execute many thread blocks concurrently. When multiple thread blocks are running concurrently on an SM, context switching between them is cost-free. This allows the hardware to hide stalls and latency by simply tracking which threads are ready to execute their next instruction and issuing instructions to whichever threads are available. The more threads the SM has to choose from, the more effectively this works. This is known as [hardware multithreading](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#hardware-multithreading), and many older resources on CUDA performance discuss it as the primary principle for writing fast kernels.

At this stage, the factor limiting the number of thread blocks that can reside on an SM is shared memory. Each thread block is allocating a $(256,64)$ tile of shared memory for the $A$ matrix and a $(64,128)$ tile for the $B$ matrix. This amounts to 49 KB out of the total 64 KB of shared memory per SM, which restricts the number of resident thread blocks on an SM to one at a time. Therefore, in this situation, since shared memory is the limiting factor, using more registers per thread does not have a negative impact.

High-performance GEMM kernels often have lower occupancy, meaning they utilize more shared memory and register memory per thread and have fewer threads resident on an SM at once. This is primarily due to the need for high arithmetic intensity; to keep compute units busy with limited memory bandwidth, more computation per thread at lower levels of the memory hierarchy is better. However, the disadvantage of low occupancy is that the GPU will be less effective at automatically hiding latency through context switching. We can manage this trade-off by structuring our kernel to allow for overlap between computation and data movement, as this chapter exemplifies.

The two most recent NVIDIA architectures, Ampere and especially Hopper, have introduced dedicated hardware support that allows us to perform several components of GEMM kernels asynchronously. This hardware support makes writing efficient, low-occupancy kernels like these much more straightforward.

## Kernel 5 - Tune Tile Dimensions
Up to this point, I consistently found that after about 10 minutes of examining the profiling results in NSight Compute, I could pinpoint the exact bottleneck in the kernel and its cause. After developing Kernel 4, which reached about 70% of cuBLAS throughput, the profiler generally no longer pointed to a single, clear performance issue. In retrospect, this was because the remaining 30% gap between Kernel 4 and cuBLAS was the result of many smaller inefficiencies rather than one major one, and performance optimization started to take on a more trial-and-error character based on hunches, some of which proved to be incorrect. This chapter describes two optimizations that, when implemented together, produced a respectable speedup.

### tune tile dimensions

At this point, I started to wonder, if my kernel was still memory-bound, how would I be able to tell? If you are using single-precision FFMA instructions, the "Speed of Light" section in NSight Compute displays a roofline chart, but this is not the case if you are using tensor cores. I was motivated by [this](https://www.cse.ust.hk/~weiwa/papers/yan-ipdps20.pdf) paper to try to figure it out myself using a back-of-the-envelope calculation.

A more practical way to frame the "am I memory-bound?" question is "does my kernel's arithmetic intensity surpass the machine's balance point?"

$$ \frac{FLOPs\ performed}{bytes\ moved} \stackrel{?}{>} \frac{\tau}{\beta} $$

So, for the left side of the inequality, we need to plug in values specific to our kernel, and for the right side, we need to substitute values specific to our hardware. The [section on arithmetic intensity](#arithmetic-intensity-as-a-function-of-tile-dimensions) explained how arithmetic intensity is a function of tile sizes. Specifically, for tile dimensions $BM,BN$ and $BK$, the expected arithmetic intensity is $\frac{BM * BN}{BM+BN}$. Here is a reminder of this, focused on the block tile level:
![intensity_block_tile_dims](/images/intensity_block_tile_dims.png) 
Note how $BK$ is eliminated in this calculation. This implies that when considering arithmetic intensity, the size of our tiles along the $K$ dimension is irrelevant. However, when thinking about other aspects of performance, it is not irrelevant (more on this later).

#### M and N Dimensions / L2 cache locality
We now need to substitute in the values for our machine's balance. Earlier in the roofline charts, we set $\tau_{HMMA}$ to the throughput of the cuBLAS hgemm kernel, which likely tends towards being an underestimate. In this instance, the objective is to select tile dimensions large enough to place us comfortably within the compute-bound region of the roofline chart, so I would prefer to err on the side of overestimating the arithmetic throughput in the numerator of the machine balance and underestimating the memory bandwidth in the denominator.

A reasonable overestimation for $\tau_{HMMA}$ is 65,000 GFLOP/sec, which is the theoretical peak performance found on the T4's data sheet.

When considering the memory bandwidth in the denominator, we want to conservatively estimate our achieved memory bandwidth. To do this, we must take into account the effect of the L2 cache. The L2 cache is shared among the 40 streaming multiprocessors on the T4. In practice, this means that when one thread block accesses data from DRAM, that data is moved into the L2 cache, and subsequent accesses to the same data from other thread blocks will hit the L2 cache, until that piece of data is eventually evicted.

According to [individuals on the internet](https://stackoverflow.com/questions/46660053/is-blockidx-correlated-to-the-order-of-block-execution), thread blocks are executed in increasing order of their flattened block index. The official CUDA programming guide states that different thread blocks execute independently and that the programmer should not make any assumptions about the relationships between them. So, relying on this assumption for correctness would likely be ill-advised, but for a quick and approximate calculation of L2 cache locality, it is helpful.
![l2_cache_locality](/images/l2_cache_locality.png)
The fundamental concept here is that accesses to the $A$ matrix from thread blocks executing concurrently exhibit much better locality than accesses to the $B$ matrix. The majority of accesses to $A$ should hit the L2 cache, whereas most accesses to $B$ are likely to miss, which means we should achieve roughly a 50% hit rate for global memory accesses. This implies that our *achieved* memory bandwidth is a 50/50 weighted sum of the DRAM bandwidth and our L2 cache bandwidth. Substituting this weighted sum into the denominator of the expression for machine balance finally gives us:

$$ \frac{BM * BN}{BM+BN} \stackrel{?}{>} \frac{\tau_{HMMA}}{0.5 * \beta_{DRAM} + 0.5 * \beta_{L2}} $$

Plugging in the current block tile dimensions ($BM=256$ and $BN=128$), memory bandwidths, and theoretical arithmetic throughputs gives us:

$$ \frac{256 * 128\ FLOPs}{256 + 128\ bytes} \stackrel{?}{>} \frac{65,000 * 10^9\ FLOPs/sec}{0.5 * 220 * 10^9 + 0.5 * 1280 * 10^9\ bytes/sec} $$

This calculation results in an arithmetic intensity of $85.3 \frac{FLOPs}{byte}$ and a machine balance of $87.24 \frac{FLOPs}{byte}$. The fact that these two figures are very close suggests that global memory access may still be the dominant factor in our overall runtime. If we can afford the space in shared memory, increasing our $BN$ dimension from 128 to 256 might be beneficial. If both $BM$ and $BN$ are 256, our estimated arithmetic intensity rises to $128.0 \frac{FLOPs}{byte}$, which should hopefully place us comfortably in the compute-bound regime.

When we look at the next level down in the hierarchy, the high shared memory bandwidth provides us with a bit more leeway. Our swizzled shared memory layouts should lead to bank-conflict-free access, giving us the full bandwidth of 3662 GB/sec. The $WM$ and $WN$ dimensions of the warp tiles are both 64. Plugging these numbers into the formula:

$$ \frac{WM * WN}{WM+WN} \stackrel{?}{>} \frac{\tau_{HMMA}}{\beta_{shmem}} $$

yields an arithmetic intensity of $32 \frac{FLOP}{byte}$ and a balance point of $17.7 \frac{FLOP}{byte}$. It is therefore safe to assume that shared memory loads are not the primary bottleneck in our kernel's runtime. However, to err on the side of greater arithmetic intensity, I also ended up increasing $WM$ and $WN$ while reducing $WK$.

#### K Dimension
Different factors come into play when we consider our tile sizes along the K dimension. In our on-paper analysis, the tile size along the K dimension cancels out of the expression for arithmetic intensity. When we think about tile lengths along this dimension, other considerations become relevant. First, we can use it to modify the total size of our tiles without impacting arithmetic intensity. In the case of block tiles, the total number of bytes of shared memory they occupy is $BK* (BM+ BN)* sizeof(half)$, so increasing $BK$ by one unit increases the total size of the block tiles by $(BM+ BN)* sizeof(half)$. When deciding the length of the block tiles along the K dimension, this becomes the primary consideration. With $BN=256$ and $BM=256$, we select $BK=32$; with these dimensions, the total amount of shared memory used by the tiles of $A$ and $B$ comes to 32 KiB, which is exactly half of the shared memory available per streaming multiprocessor. This choice makes sense in the next section, which introduces a technique called shared memory double buffering. This optimization involves allocating two buffers in shared memory for each input matrix, allowing one to be written to while the other is being read. When double buffering is implemented with these tile dimensions, we will be using every available byte of shared memory on the device.

### tile dimensions - longer and thinner
Here is a visualization of the before and after states:
![tile_dims_adjustment.png](/images/tile_dims_adjustment.png)
Both the block tiles and warp tiles are made longer and narrower along the K dimension to boost arithmetic intensity. For the sake of time, I combined this optimization with the optimizations discussed below, so I did not measure the performance improvement of this change in isolation.

## Kernel 5 - Optimize Index Calculation
At this stage, I was at about 70% of cuBLAS performance, and my primary strategy for using NSight Compute was to compare kernel metrics between my own kernels and the cuBLAS HGEMM kernel. Although the source code for the cuBLAS HGEMM implementation is not public, examining its metrics collected by NSight Compute can provide insights into the kinds of optimization techniques the clever engineers at NVIDIA might have employed when developing it.

The one thing that immediately caught my attention was that the total number of executed instructions for the cuBLAS HGEMM was $94,175,232$, whereas Kernel 4 was executing $216,227,840$, more than twice as many instructions compared to cuBLAS. While Kernel 4 partially makes up for this by having a lower cycles-per-instruction (CPI) ratio (around 8, versus about 12 for cuBLAS), this discrepancy is certainly worth investigating.

So I began to wonder, why is my kernel executing twice as many instructions? Expanding the instruction mix section in NSight Compute provides more information.
![instruction_mix_comparison](/images/instruction_mix_comparison.png)
The reason is that Kernel 4 is performing significantly more index calculation-related instructions than the cuBLAS kernel. The `LOP`, `IADD3`, and `SHF` instructions are integer and logical operations; these utilize different pipelines from the tensor core and can execute concurrently with floating-point math happening elsewhere on the chip. However, each warp scheduler on a streaming multiprocessor can only issue a single instruction per cycle, so the large volume of index calculation instructions is likely displacing the issuance of the `HMMA` instructions, which are the tensor core instructions doing the heavy computational work. So, what are these integer and logical instructions doing, and why are there so many of them?

According to NSight Compute, 92% of the total instructions executed by Kernel 4 are within the loop nest where each warp loads its region of data from shared memory into register memory, and then performs an outer product over local matrices stored in register memory using a series of `HMMA` instructions. The three nested loops that map the `HMMA` instructions to their positions are all fully unrolled, so there is no runtime index calculation needed there.

However, the `HMMA` instructions operate on $8$ by $8$ tiles stored in registers, and before the compute phase, the threads in each warp work together to load all of these tiles from swizzled shared memory into register memory using the `ldmatrix` PTX instruction (see [here](#how-to-use-tensor-cores) for an explanation of `ldmatrix`). Since we are at the very bottom of the tile hierarchy at this point, the tiles are extremely small, and as a result, we are performing this index calculation *many* times ($O(\frac{N^3}{8})$). This calculation involves multiplying by several strides, computing a modulo with respect to the thread index, and several logical operations to apply the swizzling function, all of which occurs at runtime.

![index_calculation_inneficient](/images/index_calculation_inneficient.png)

To make this more efficient, we should move as much of this calculation as possible to be performed at compile time, and any part that must happen at runtime should be as streamlined as possible. In the index calculation code shown above, there are fundamentally three distinct and dependent steps:
1. First, each warp computes the memory address of the top-left corner of the MMA tile.
2. Next, each thread calculates the memory address of the element it will load, relative to the address from step (1).
3. Finally, because our shared memory layout is swizzled, each thread applies the swizzle function to the address computed in step (2) to get the correct memory address in the swizzled layout.

All three of these steps are executed for each of the 8x8 MMA tiles. Below is a visualization of this process; the diagram shows a mini-example where each MMA tile is four rows and one column, and each warp tile contains 2x8 MMA tiles (using simpler examples like this allows us to make all the details as explicit as possible, and the devil is in the details).

![swizzled_index_calculation_inneficient](/images/swizzled_index_calculation_inneficient.png)

In the middle column, each thread has determined the address of the value it will load, based on the unswizzled layout. In each iteration, these pointers are advanced to the right by one column, until we reach the end of the warp tile, at which point we move down to the next set of rows. If it were not for the swizzled layout, we could simply advance the pointers by one in each iteration, i.e., `thread_row+=1`. However, because the data is stored in a swizzled layout, advancing the pointers to the next group of MMA tiles is not as simple as incrementing by one.

While incrementing by one will not suffice for iterating through a swizzled layout, we can achieve the same effect by XORing each thread's pointer with a constant.
![swizzled_index_calculation_efficient](/images/swizzled_index_calculation_efficient.png)
This optimization reduces the amount of index calculation from approximately 13 operations between each `ldmatrix` call down to a single XOR. After applying this change, the total number of executed instructions drops to around 90 million, which is slightly less than that of cuBLAS.

This illustrates the fundamental principle of iterating efficiently through a swizzled data layout. In the [actual code](https://github.com/alexarmbr/matmul-playground/blob/main/src/kernel5.cu#L10), the process is a bit more complex because the swizzle function is more intricate, and we need to iterate through the tiles of A and B, which have different dimensions. Additionally, the loops containing the `ldmatrix` instructions are manually unrolled; this simplifies the XORing process and may also enable the compiler to do a better job of interleaving the `ldmatrix` and `mma.sync` instructions to balance the load between the two different hardware pipelines.

The optimized index calculation, loop unrolling, and adjusted tile dimensions are all implemented as part of the same kernel, which achieves a hard-won 1.2x speedup over the previous one and brings us to 86.7% of cuBLAS throughput.
![table5](/images/table5.png)

## Kernel 6 - Double Buffering
Back to the profiler we go (for the final time). At this point, many of the metrics for my kernel and cuBLAS were beginning to look quite similar. One thing that stood out to me was that the threads in my kernel were spending more time stalled on `__syncthreads()` than those in the cuBLAS kernel. At this stage, my kernel had a CPI (cycles per instruction) of 14, and about 2.6 of these cycles were due to synchronization stalling. So, while this was not a catastrophic performance issue, it was noticeable. A technique known as double buffering allows you to eliminate one of the two `__syncthreads()` calls in the inner loop. After some consideration, I realized that this offers no guarantee of a proportional reduction in cycles stalled on `__syncthreads()` (if you remove one `__syncthreads()`, threads might just spend twice as long stalled on the other). However, double buffering should also permit a bit more instruction-level parallelism within the main loop, it is implemented in CUTLASS kernels, and I had the shared memory to spare, so I thought, why not?

The data dependencies within the main loop of our current GEMM kernel require two `__syncthreads()` calls to prevent race conditions in shared memory.
![two_syncthreads](/images/two_syncthreads.png)
If we were to remove either of these, race conditions would arise because the mapping of threads to data is different for writing to shared memory versus reading from it. This is because any given thread is performing computations on different values than the ones it fetched from global memory and wrote to shared memory. Consequently, synchronization points are necessary to prevent race conditions, as the entire thread block must wait until all threads have finished writing to shared memory before any thread can begin reading from it.

The drawback of these synchronization points is reduced parallelism and potentially lower hardware utilization. As the diagram above indicates, the main loop consists of four primary components:
1. Prefetching the next block tile into registers.
2. Transferring data from shared memory to registers in preparation for computation.
3. The computation itself.
4. Writing the prefetched data back from registers to shared memory.

As the diagram illustrates, component #4 is kept separate from the other three because it involves writing to the same data that is being read in #2; that is, all 256 threads in a block must finish #2 before any can start #4. This separation is detrimental to performance because it constrains the compiler's ability to interleave instructions of different types to balance the load across various hardware pipelines.

The concept behind double buffering is that if we allocate an additional pair of shared memory buffers for the block tiles of $A$ and $B$, we can write to one pair of buffers while the other is being read concurrently. This enables us to remove the second `__syncthreads()` from the main loop, which should result in a slight performance improvement.

![one_syncthreads](/images/one_syncthreads.png)

The two changes made here are the removal of one of the `__syncthreads()` calls and the addition of an index that we always use (`%2`) to keep track of which of the two buffers is being read and which is being written in any given iteration. The buffer being read and the buffer being written switch with each iteration.

![double_buffering](/images/double_buffering.png)

This results in a minor speedup over the previous kernel. But at this stage of trying to optimize an already highly tuned kernel, I will accept whatever gains I can get.

![table_6](/images/table6.png)

# Conclusion
## things I didn't do
And this is the point where I decided to wrap things up! There are two potential avenues for further performance enhancements, but the time I had allocated for this project ran out. The first of these is significantly easier than the second.
- **optimized epilogue** - As a reminder, the GEMM problem is defined as $D=\alpha * A * B + \beta * C$. This single kernel actually performs two computations. The majority of the compute is in the matrix multiplication $C^\*=A * B$. After we multiply the two matrices, we then calculate $D = \alpha * C^* + \beta * C$, which is generally known as the kernel epilogue. The former is an $O(N^3)$ problem, while the latter is $O(N^2)$. When N is large, the matrix multiplication part dominates the runtime of the combined algorithm, but when N is smaller, the epilogue becomes more significant. This article focused entirely on the matrix multiplication, as it is the most interesting and critical component of the GEMM problem. The kernel epilogue I used in all six kernels is inefficient—once the matrix multiplication is finished, the result is scattered across thread registers according to the `m16n8k8` MMA layout (see [below](#appendix)), and then written directly back to memory. This write operation is uncoalesced and therefore achieves less than ideal bandwidth and latency. Improving this would likely reduce the performance gap between Kernel 6 and cuBLAS for smaller matrix sizes.
- **manual instruction mix tuning for inner loop** - Projects like [this one](https://github.com/NervanaSystems/maxas/wiki/SGEMM) and [this one](https://github.com/daadaada/turingas) match or even exceed the performance of cuBLAS by using custom-built assemblers that allow them to write the entire kernel in SASS. The inner loop of a GEMM kernel is composed of shared memory loads and math instructions. If too many instructions of one type are clustered together, hardware pipelines can become overloaded, leading to stall cycles. If you choose to write your kernels entirely in CUDA and PTX, as I did, then instruction scheduling is handled by the compiler; the fact that I was able to achieve over 90% of cuBLAS performance without any inlined assembly suggests that `nvcc` likely does a pretty good job of it. However, if one were truly set on writing a kernel that is as fast as or faster than cuBLAS for a range of matrix sizes, this approach would probably be necessary.

## performance on different matrix sizes
Here is a chart that displays the performance of the kernels I developed, compared to cuBLAS, across a variety of matrix dimensions.
![hgemm_performance](/images/hgemm_performance.png)

It's worth noting that the performance gap between the fastest kernel I wrote and the cuBLAS HGEMM is slightly wider for smaller matrices, possibly due to my unoptimized epilogue. It is also possible that this is because cuBLAS is selecting kernels that have been specifically tuned for those particular matrix dimensions.

## lessons learned, newer GPUs are better
Given the number of people and companies currently purchasing NVIDIA GPUs almost solely for the purpose of running matrix multiplications, it seems a great deal of effort goes into enhancing the tensor cores in terms of programmability and performance with each successive architecture. The tensor core throughput increases by an order of magnitude with each new SM architecture, and while memory bandwidth also improves, it does not do so proportionally.

To make the task of programming these powerful yet imbalanced machines more manageable, the more recent Ampere and Hopper architectures have introduced hardware support that allows several critical parts of a GEMM kernel to run asynchronously with respect to the rest of the SM. Ampere introduced hardware support for [asynchronous data copying](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html#asynchronous-data-copy-from-global-memory-to-shared-memory) from global memory to shared memory; I implemented a sort of makeshift version of this using extra registers in [Kernel 4](#kernel-4---makeshift-async-copy). The Hopper architecture introduced something even more advanced called the [Tensor Memory Accelerator](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator), which is essentially a copy engine capable of performing index calculation and initiating global memory transfers asynchronously from the rest of the SM. Consequently, developers writing kernels for Hopper likely do not need to worry about the efficiency of index calculation (as we did [here](#kernel-5---optimize-index-calculation)), because this task is offloaded to dedicated hardware in the TMA. Hopper also features asynchronous tensor core instructions that can read from and write to shared memory, rather than registers (see [here](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)).

All of this asynchronicity is a tremendous benefit for low-occupancy, register-intensive GEMM kernels. As discussed [here](#gpu-occupancy-digression), high arithmetic throughput requires a large amount of fast memory to cache data, which means we cannot run as many threads per SM. This, in turn, means the GPU will not automatically hide our latency through context switching, forcing us, the programmers, to think more carefully about how our latency is being concealed. This is where asynchronicity proves to be helpful.

All of this indicates that Hopper is a somewhat new and different kind of machine; if you examine GEMM kernels in CUTLASS that target Hopper, you will notice the code has a different structure compared to all the pre-`sm_90` kernels. Hopper kernels employ a producer-consumer pattern, where a relatively small number of producer threads initiate asynchronous data copies with the TMA, and then consumer threads manage the tensor cores. I have never worked on kernels targeting Hopper, so my knowledge on this is limited at the moment; [this](https://hazyresearch.stanford.edu/blog/2024-05-12-tk) article offers an interesting overview of the user experience of writing kernels for Hopper.

This is all to say that the kernels discussed here are designed for the Turing architecture, which was state-of-the-art in 2018, and if you are writing kernels for Ampere or Hopper, the techniques you use for latency hiding will be different and simpler. I used the Tesla T4 GPU because it can be rented on AWS for about 50 cents per hour, which is about as much money as I am willing to spend on EC2 instances. Using an older GPU for this project was both a blessing and a curse; the curse was that no special hardware support was available for hiding memory latency or calculating indices, while the blessing was that I had to implement all of this myself, which was a valuable educational experience!

# Are you hiring GPU nerds?
I am typically not one for self-promotion, but I recently took a short break from work and am now back on the job market. If you are a hiring manager seeking someone to tinker with kernels, profilers, and/or compilers, please feel free to email me!