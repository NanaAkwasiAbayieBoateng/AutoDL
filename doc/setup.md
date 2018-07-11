
###  setup:
    0. build tensorflow depency
       apt install libibverbs-dev
       apt-get install cuda-command-line-tools-9
       export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64
       

       apt-get install --reinstall python3-minimal python-lockfile
       apt-get install python3-numpy python3-dev python3-pip python3-wheel
       
       #missing input file '@local_config_nccl//:nccl/NCCL-SLA.txt
       # cp to third_party/nccl/NCCL-SLA.txt
       
       bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
       bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
       pip3 install -U tensorflow-1.8.0-cp35-cp35m-linux_x86_64.whl 

 

       pip3 install -U pyyaml
       pip3 install -U torch
    
       pip3 uninstall tensorflow-gpu
       pip3 uninstall tensorflow
       pip3 install  --no-cache-dir -U tensorflow-gpu
    

    1. install nccl2: 
        perfer : NCCL 2.1.0, this issue is no longer present when using CUDA 9,
        https://docs.nvidia.com/deeplearning/sdk/nccl-install-guide/index.html
  
    2. install openmpi3.0:
        wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz && \
        tar zxf openmpi-3.0.0.tar.gz && \
        cd openmpi-3.0.0 && \
       ./configure --enable-orterun-prefix-by-default && \
        make -j $(nproc) all && \
        make install && \
        ldconfig && \
   
    3. install horovod
        # fix build error: : fatal error: tensorflow/compiler/xla/statusor.h: No such file or directory
        /usr/local/lib/python3.5/dist-packages/tensorflow/include/tensorflow# cp -r /tensorflow/tensorflow/compiler  .

        sudo -E HOROVOD_GPU_ALLREDUCE=NCCL  HOROVOD_WITH_TENSORFLOW=1 pip install --no-cache-dir -U horovod


    4. some configure:
        # Configure OpenMPI to run good defaults:
        # --bind-to none --map-by slot --mca btl_tcp_if_exclude lo,docker0
        echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf && \
        echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf && \
        echo "btl_tcp_if_exclude = lo,docker0" >> /usr/local/etc/openmpi-mca-params.conf
    
    5. # Set default NCCL parameters
        echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
        echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf
    6. for nvidia docker 
        --shm-size=1g --ulimit memlock=-1

    



###  usage:
   mpirun -np 4 \
    -H server1:1,server2:1,server3:1,server4:1 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
    cifar10_train.py [cifar10_train.yaml]

### TODO: 
 - save checkpiont by params
