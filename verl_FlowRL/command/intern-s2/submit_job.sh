#!/bin/bash
# FlowRL Qwen2.5-7B Math Training Job Submission Script

# Job configuration
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
JOB_NAME="flowrl-qwen25-7b-math-v0-4-0-${TIMESTAMP}"
GPU_COUNT=8
MEMORY=1600000
CPU_COUNT=128
CHARGED_GROUP="llmit_gpu"

rjob submit \
--name=${JOB_NAME} \
--gpu=${GPU_COUNT} \
--memory=${MEMORY} \
--cpu=${CPU_COUNT} \
--charged-group=${CHARGED_GROUP} \
--private-machine=group \
--mount=gpfs://gpfs1/llmit:/mnt/shared-storage-user/llmit \
--mount=gpfs://gpfs1/large-model-center-share-weights:/mnt/shared-storage-user/large-model-center-share-weights \
--mount=gpfs://gpfs1/llmrazor-share:/mnt/shared-storage-user/llmrazor-share \
--image=registry.h.pjlab.org.cn/ailab-puyu/xpuyu:torch-2.7.0-076676dd-0708 \
-P 8 \
--gang-start=true \
--host-network=true \
-e DISTRIBUTED_JOB=true \
--custom-resources rdma/mlnx_shared=8 \
--custom-resources mellanox.com/mlnx_rdma=1 \
-- bash /mnt/shared-storage-user/llmit/user/xuekaizhu/FlowRL/verl_FlowRL/command/intern-s2/run_flowrl.sh
