### tf-multigpu 
- A tf train framework for easy and as much as GPU use ratio.

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



### impelement
 - replicated as GPU -> doing
 - add model zoo     -> doing

### profile
- config.inter_op_parallelism_threads : set to cpu core num 
