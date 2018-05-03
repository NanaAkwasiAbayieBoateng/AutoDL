###  brocaster

### motivation
- As SGD distribute training, the key problem is how to sync grad aross multi worker.
  this work focus mutli machine synchronization data, Two optimize:
    1. The bandwidth Utilization
    2. The lantency when reduced 

### the GPU topology 
```
              |           |
             [root] <-> [root]
              |     \ /   |
              |     / \   |
            [ leaf ]   [ leaf ]     ---- (up  grid)
            | | | |    | | | |
            N N N N    (N N N N)   ----- (sub grid)
```
- As many data centor or hpc use FatTree as topo, although they have multi-link redundancy
  There are many allreduce algorithm, See also : [allreduce_halving_doubling](https://github.com/facebookincubator/gloo/blob/master/docs/algorithms.md#allreduce_halving_doubling)
- Now use hierarchy reduce:
    1. reduce in (sub grid): local rank 0 as master gatter and sum, assume Rank0 has to P2P to other
    2. reduce in (up grid):  local rank 0 as master gatter and sum
    3. recursion to root, then backwise to sub grid


####  how to used bandwidth Utilization:
- TCP: as keep many connection send same time
- UDP: use UDP, but need to implement crc and resend.

#### how to keep low lantency
- DPDK: TODO :  seastar & DPDK have a test, as netmap, PF_RING DNA has little github to copy.
  - https://github.com/F-Stack/f-stack & http://www.f-stack.org/
  - seastar: https://github.com/scylladb/seastar
  - ans: https://github.com/ansyun/dpdk-ans
- RDMA: 
  - DPDK vs RDMA : see https://www.jianshu.com/p/09b4b756b833?utm_campaign
  - softRoce:
  - RoceV2 :

### timeline

``` 
    reduceop  |  shared_map | shared_queue |  MessagerA      
  ---->|
       | -tensor- push ->|
       |       ---push------------>|          ^-Loop 
       |                                      ^  | -> send_tensor-> B 
       |                                      ^  | -> send_tensor-> C
       |                                      ^  | ...
       |  <---------------------notify--------^- |  on wait B,C ..
 done<-|
``` 

### op & done design
- brocaster.init(iplist, ports, channel=seastar_tcp, channel_num=10): 
   - setup every channel
   - build local master
   - setup local shared memery

- brocaster.brocast(tensor, step): send tensor to master 
- brocaster.allreduce(tensor, step): recevie tensor from master



