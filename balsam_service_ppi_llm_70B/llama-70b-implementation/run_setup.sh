#!/bin/bash
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate

#source /home/atanikanti/anaconda3/bin/activate
conda create --prefix /lus/gila/projects/Aurora_deployment/atanikanti/environments/llama-70B-ppi-llm-env python=3.9 --y
conda activate /lus/gila/projects/Aurora_deployment/atanikanti/environments/llama-70B-ppi-llm-env

module unload oneapi
conda install cmake ninja --y
cd /lus/gila/projects/Aurora_deployment/atanikanti/70B-opt/
pip install -r torch_requirements.txt
pip install torch-2.0.0a0+gitd475571-cp39-cp39-linux_x86_64.whl 

module use -a /home/ftartagl/modulefiles
module load oneapi-testing/2023.2.003.PUBLIC_IDP49422

git clone  https://github.com/huggingface/transformers.git
cd transformers
git clone e42587f596181396e1c4b63660abf0c736b10dae
python setup.py develop
cd ..


pip install huggingface_hub sentencepiece accelerate tqdm pandas
conda install mpi4py --y
conda uninstall --force mpich --y

pip install -r ipex_requirements.txt
pip install intel_extension_for_pytorch-2.0.110+git340debe-cp39-cp39-linux_x86_64.whl 

module load oneapi-testing/2023.2.003.PUBLIC_IDP49422
cd frameworks.ai.pytorch.torch-ccl
pip install -r requirements.txt
COMPUTE_BACKEND=dpcpp python setup.py develop
mkdir -p oneccl_bindings_for_pytorch/bin/temp
mv oneccl_bindings_for_pytorch/bin/* oneccl_bindings_for_pytorch/bin/temp

find  oneccl_bindings_for_pytorch/bin -type f -exec  mv {}  oneccl_bindings_for_pytorch/bin/temp \;

mkdir -p oneccl_bindings_for_pytorch/lib/temp
find  oneccl_bindings_for_pytorch/lib -type f -name "*fabric*" -exec mv {}  oneccl_bindings_for_pytorch/lib/temp \;

find  oneccl_bindings_for_pytorch/lib -type f -name "*mpi*" -exec mv {}  oneccl_bindings_for_pytorch/lib/temp \;
#mv oneccl_bindings_for_pytorch/lib/*mpi* oneccl_bindings_for_pytorch/lib/temp

#mv oneccl_bindings_for_pytorch/lib/*fabric* oneccl_bindings_for_pytorch/lib/temp
cd ..



git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
git checkout 8c1eed2e4750d1f8e963f5e85ff38f9053072e46
pip install -r requirements/requirements.txt
cd ..

git clone https://github.com/intel/intel-extension-for-deepspeed.git
cd intel-extension-for-deepspeed
git checkout aad672b49931f4a7e5516518703491e0849e324a
python setup.py develop

cd ../DeepSpeed
python setup.py develop

cd ..
