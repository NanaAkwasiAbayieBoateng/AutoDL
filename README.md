### AutoML(0.1)
- A tf train framework for auto train the nice mode.
- This project now is developing, now as code showing

### Feature
- sample API & module: keep sample & keep easy to replace.
- multi GPU support: 
  - replcated for P2P with GPU such as NVLINK
  - BlacePlament for low latency and limit bandwith such as PCI-E*16 


### **TODO**:
  1. **distribute support**: horovd(MPI) and broadcast_proxy.
     - horovd (MPI & Roce)
     - broadcast_proxy： seastear, DPDK
  2. **Auto hyterparam select**: detect the cpu, gpu, network limits, as much use evey resource

  3. **Model zoo & Data zoo**: shared state of art some models.


### grad variable sync strategy
- multi machine
   - allreduce(RDMA + MPI) > RDMA + grpc > parameter server
- multi GPU
   - For GPU can P2P communication use replicated at GPU
   - For GPU connect to different cpu or differnet PCI-E switch use relicated at CPU
- split variabley
   - if max size too large split it to slice (some like model parallel)

### pipeline and streaming 
- keep buffer with pipeline to keep every step max concurrency
```
   Data -> [stageArea]-> GPU0
                      -> GPU1 -> grad -> synchronization -> avg -> apply       |
                              -> back Propagation                        |-> forward 
```



### impelement API
 1. replicated as GPU
   - keras parameter store cpu,  multi-gpu apply each mode
 2. add model zoo


### profile tips
- config.inter_op_parallelism_threads : set to cpu core num
- 

### devices profile
 - Storage device:
      - SATA3.0: 6GB/s bandwith
      - M.2 (socket2->pcie2 or sata:500MB/s, socket3: pcie3 * 4: 32Gb/s)
      - NVMe(Non volatile Memory Express) pci-e
          - IOPS: 4k > 3WIOPS
          - bandwith: 4K*64thread > 1GB/s ;
          - access lantency: 0.02 ms
          
  - ZMQ: maybe a good choice for multi machine    
