### AutoML(0.2)
- A tf train framework for auto train the nice mode.
- This project now is developing, now as code showing

### performance
- ON single 8-V100, 936 per sec at imagenet batch 1024

### Feature
- sample API & module: keep sample & keep easy to replace.
- multi GPU support: 
  - replcated for P2P with GPU such as NVLINK
  - BlacePlament for low latency and limit bandwith such as PCI-E*16 

### TODO cuda mem_setop
- 
- 

### **TODO**:
  1. **distribute support**: horovd(MPI) and broadcast_proxy.
     - horovd (MPI & Roce)
       - each a singe process 

     - broadcast_proxy： seastear, DPDK
  2. **Auto hyterparam select**: detect the cpu, gpu, network limits, as much use evey resource
     - detect system configure
     - train and eval speed measure
  3. **Model zoo & Data zoo**: shared state of art some models.
     - imagenet

###　Benchmark
  - DGX workstation: V100*4, NVLink 100GB, 1100 sample/sec for imagenet resnet50 batchsize:128