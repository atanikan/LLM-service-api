#!/bin/bash

#PBS -A Aurora_deployment
#PBS -q workq
#PBS -l select=1
#PBS -l walltime=00:60:00

module unload oneapi
#module load gcc/12.1.0
export HF_HOME=/home/jmitche1/huggingface
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export ENABLE_SDP_FUSION=1
export COL_MAJOR=0

export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
export CCL_OP_SYNC=1
export CCL_PROCESS_LAUNCHER=pmix
export FI_PROVIDER=cxi
export PALS_PMI=pmix
export CCL_ATL_TRANSPORT=mpi # Required by Aurora mpich
export FI_MR_CACHE_MONITOR=disabled # Required by Aurora mpich (HPCS-6501)
export CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=32768
export I_MPI_ROOT=/opt/cray/pe/pals/1.2.12/bin/mpiexec

source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate

conda activate /lus/gila/projects/Aurora_deployment/atanikanti/environments/openai-ppi-llm-env

module use -a /home/ftartagl/modulefiles
module load oneapi-testing/2023.2.003.PUBLIC_IDP49422


source /lus/gila/projects/Aurora_deployment/atanikanti/70B-opt/frameworks.ai.pytorch.torch-ccl/third_party/oneCCL/build/_install/env/vars.sh

cd $PBS_O_WORKDIR

echo $PROTEIN

if [ ! -d DIR.$PROTEIN ]; then
  mkdir DIR.$PROTEIN
fi

cp prompt_llama_70B.py DIR.$PROTEIN
cp proteins.csv DIR.$PROTEIN
cd DIR.$PROTEIN

for i in {1..10}
do
    echo $i
    echo $PROTEIN
    python ./prompt_llama_70B.py $PROTEIN > $i.output.txt;
    wc -l $i.output.txt
    wc -l prompt_llama_70B.output.txt
    mv prompt_llama_70B.output.txt $i.long.output.txt;
done



#mpirun -np 4 python -u run_generation_with_deepspeed.py -m /home/jmitche1/huggingface/llama2 --ipex --benchmark --input-tokens=1024 --max-new-tokens=128 |& tee llma-70-opt-ipex.log
