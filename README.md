# Running Llama2 Inference on Sunspot

We have provided 3 options to run the Llama LLM on Sunspot - optimized by Intel to run on PVC.

* 13B and 70B Llama2 model using bash scripts on Sunspot
* 13B and 70B Llama2 model using Parsl
* 13B and 70B Llama2 model using REST API which wraps parsl from your laptop/desktop (Work in Progress)

# Table of Contents
- [13B Quick Start](#13BQuickStart)
- [70B Quick Start](#70BQuickStart)
- [LLAMA2 Files, Data and Scripts](#LLAMA2Files)
- [Sunspot](#Sunspot)
- [Building a custom 13B conda environment](#13BBuildingEnvironments)
- [Building a custom 70B conda environment](#70BBuildingEnvironments)
- [Inference with Parsl](#Inference_with_Parsl)
- [Restful Inference Service - WIP](#Restful_Inference_Service)
- [Globus Compute/FuncX Service](#funcX)

<a name="13BQuickStart"></a>
## 13B Llama2 Inference - Quick Start Guide

1. SSH to sunspot
```
ssh -J username@bastion.alcf.anl.gov username@sunspot.alcf.anl.gov
```

2. Sample submission script  
:bulb: **Note:**  This requires only 1 tile to run
You can find this script at: `/lus/gila/projects/Aurora_deployment/anl_llama/submit_scripts/submit_13b.sh`

```bash
#!/bin/bash

#PBS -A Aurora_deployment
#PBS -q workq
#PBS -l select=1
#PBS -l walltime=30:00

export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 
export ENABLE_SDP_FUSION=1

source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate
#This is the conda environment for the 13B llama inference
conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-13b

source /soft/compilers/oneapi/2023.05.15.001/oneapi/compiler/latest/env/vars.sh
source /soft/compilers/oneapi/2023.05.15.001/oneapi/mkl/latest/env/vars.sh

MODEL_DIR=/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/llma_models/llma-2-convert13B
SRC_PATH=/lus/gila/projects/Aurora_deployment/anl_llama/13B/intel-extension-for-pytorch/examples/gpu/inference/python/llm/text-generation

#13B 32 in 32 out
#python -u $SRC_PATH/run_llama.py --device xpu --model-dir $MODEL_DIR --dtype float16 --ipex --greedy

#13B 1024 in 128 out
python -u $SRC_PATH/run_llama.py --device xpu --model-dir $MODEL_DIR --dtype float16 --ipex --greedy  --input-tokens 1024 --max-new-tokens 128

```

Next, submit the above script
```bash
qsub <foo.sh>
```

Note: This uses a conda environment located at `/lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-13b`

<a name="70BQuickStart"></a>
## 70B Llama2 Inference - Quick Start Guide
1. SSH to sunspot
```
ssh -J username@bastion.alcf.anl.gov username@sunspot.alcf.anl.gov
```

2. Sample submission script
:bulb: **Note:**  This requires 2 GPUs (4 tiles) to run
You can find this script at: `/lus/gila/projects/Aurora_deployment/anl_llama/submit_scripts/submit_70b.sh`

```bash
#!/bin/bash
#PBS -A Aurora_deployment
#PBS -q workq
#PBS -l select=1
#PBS -l walltime=120:00

export HF_HOME=/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/huggingface/llama2
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
module use -a /home/ftartagl/modulefiles
module unload oneapi
module load oneapi-testing/2023.2.003.PUBLIC_IDP49422

source /home/jmitche1/70Bccl/libraries.performance.communication.oneccl/build/_install/env/vars.sh

source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate

conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-70b

SRC_DIR=/lus/gila/projects/Aurora_deployment/anl_llama/70B/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/inference
mpirun -np 4 $SRC_DIR/run_script.sh  python -u $SRC_DIR/run_generation_with_deepspeed.py -m $HF_HOME --benchmark --input-tokens=1024 --max-new-tokens=128 
```

Next, submit the above script
```bash
qsub <foo.sh>
```

Note: This uses a conda environment located at `/lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-70b`


<a name="LLAMA2Files"></a>
## Llama2 Model Weights and Related Files
The Llama2 Model files are located at `/lus/gila/projects/Aurora_deployment/anl_llama`. You can find a .tar.gz version at `/lus/gila/projects/Aurora_deployment/anl_llama.tar.gz`
You can copy it to a directory of your choice.

```bash
tar -xvf /lus/gila/projects/Aurora_deployment/anl_llama.tar.gz -C <path>
```

<a name="Sunspot"></a>
## Getting on Sunspot 
1. SSH to sunspot
```
ssh -J username@bastion.alcf.anl.gov username@sunspot.alcf.anl.gov
```
2. Add the following to your ~/.bashrc and source it
```
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
git config --global http.proxy http://proxy.alcf.anl.gov:3128
```

<a name="BuildingEnvironments"></a>
## 13B and 70B Llama Inference using bash scripts on Sunspot

<a name="13BBuildingEnvironments"></a>
### Building a custom Conda environment for 13B model

:bulb: **Note:** You can directly use the conda environment `/lus/gila/projects/Aurora_deployment/conda_env_llm` and skip conda setup steps 1-3 below. Just run `conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-13b`

1. Create a conda module and create an environment at a location of your choice. This can be done on a login node on sunspot

```bash
conda create --prefix ~/environments/anl_llma-13b python=3.9 --y
conda activate ~/environments/anl_llma-13b
```

2. Subsequently change the file `run_setup.sh` file by removing `conda create --name anl_llma-13b python=3.9 --y` and `source activate anl_llma-13b` lines
from here `~/anl_llama/13B/run_setup.sh`. The file should now look as follows.

```bash
conda install -y libstdcxx-ng=12 -c conda-forge
conda install gperftools -c conda-forge
conda install intel-openmp
python -m pip install transformers==4.29.2 cpuid accelerate datasets sentencepiece protobuf==3.20.3
module unload oneapi
pip install -r py_requirements.txt
pip install whls/torch-2.0.0a0+gite9ebda2-cp39-cp39-linux_x86_64.whl
source /soft/compilers/oneapi/2023.05.15.001/oneapi/compiler/latest/env/vars.sh
source /soft/compilers/oneapi/2023.05.15.001/oneapi/mkl/latest/env/vars.sh
pip install -r ipex_requirements.txt
pip install whls/intel_extension_for_pytorch-2.0.110.dev0+xpu.llm-cp39-cp39-linux_x86_64.whl
pip install whls/torchvision-0.15.2a0+fa99a53-cp39-cp39-linux_x86_64.whl
pip install -r audio_requirements.txt
pip install whls/torchaudio-2.0.2+31de77d-cp39-cp39-linux_x86_64.whl
```
3. Now run the file
```bash
bash run_setup.sh
```

### Running the 13B model with a custom conda environment

1. To run 13B model. Use the appropriate environment created above `conda activate <path>/environments/anl_llma-13b`. The file should look similar to this

```bash
#!/bin/bash

#PBS -A Aurora_deployment
#PBS -q workq
#PBS -l select=1
#PBS -l walltime=30:00

#export PATH=$HOME/anaconda3/bin:$PATH

#export PATH=/soft/datascience/conda-2023-01-31/miniconda3/bin:$PATH
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1 
export ENABLE_SDP_FUSION=1


source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate
#Replace with appropriate environment
conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-13b


source /soft/compilers/oneapi/2023.05.15.001/oneapi/compiler/latest/env/vars.sh
source /soft/compilers/oneapi/2023.05.15.001/oneapi/mkl/latest/env/vars.sh


cd /lus/gila/projects/Aurora_deployment/anl_llama/13B/intel-extension-for-pytorch/examples/gpu/inference/python/llm/text-generation

python -u run_llama.py --device xpu --model-dir "/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/llma_models/llma-2-convert13B" --dtype float16 --ipex --greedy |& tee llam-13b.log

#13B 1024 in 128 out
#python -u run_llama.py --device xpu --model-dir "/home/jmitche1/llma_models/llma-2-convert13B" --dtype float16 --ipex --greedy  --input-tokens 1024 --max-new-tokens 128
```
2. Now run the file. The output should be in `run_model.sh.o<>` file. If needed, you can change prompt file found here `~/anl_llama/13B/intel-extension-for-pytorch/examples/gpu/inference/python/llm/text-generation/`
```bash
qsub run_model.sh
```

:bulb: **Note:** You can use `conda deactivate` to deactivate current conda environment

<a name="70BBuildingEnvironments"></a>
### Building a custom Conda environment for 70B model

:bulb: **Note:** You can directly use the conda environment `/lus/gila/projects/Aurora_deployment/conda_env_llm` and skip conda setup steps 1-2 below. Just run `conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-70b`

1. Similar to previous repeat the same steps by creating a new environment

```bash
conda create --prefix <path>/environments/anl_llma-70b python=3.9 --y
conda activate <path>/environments/anl_llma-70b
cd ~/anl_llama/70B/
```

2. Ensure the `run_setup.sh` looks similar to this and run `bash run_setup.sh`

```bash
#!/bin/bash
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
#conda create --name anl_llma-70b python=3.9 --y
#source activate anl_llma-70b_test
python -m pip install torch==2.0.1a0 -f https://developer.intel.com/ipex-whl-stable-xpu
python -m pip install intel-extension-for-pytorch==2.0.110 -f https://developer.intel.com/ipex-whl-stable-xpu
python -m pip install oneccl-bind-pt==2.0.100 -f https://developer.intel.com/ipex-whl-stable-xpu
pip install transformers huggingface_hub mpi4py sentencepiece accelerate 
git clone https://github.com/microsoft/DeepSpeed.git
cd DeepSpeed
git checkout 7f26bb6ae47c352efeabf52f827108c42a1a55eb
pip install -r requirements/requirements.txt
cd ..
git clone https://github.com/intel/intel-extension-for-deepspeed.git
git checkout aad672b49931f4a7e5516518703491e0849e324a
cd intel-extension-for-deepspeed
python setup.py develop
cd ../DeepSpeed
python setup.py develop
```

3. The `~/anl_llama/13B/run_model.sh` should look similar to below and run `qsub run_model.sh`. Verify the full path to the conda environment in the script
```bash
#!/bin/bash
#PBS -A Aurora_deployment
#PBS -q workq
#PBS -l select=1
#PBS -l walltime=120:00
#export PATH=$HOME/anaconda3/bin:$PATH
source ~/.bashrc
#export PATH=/soft/datascience/conda-2023-01-31/miniconda3/bin:$PATH
export HF_HOME=/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/huggingface/llama2
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
module use -a /home/ftartagl/modulefiles
module load oneapi-testing/2023.2.003.PUBLIC_IDP49422oneapi-testing/2023.2.003.PUBLIC_IDP49422


#source /lus/gila/projects/Aurora_deployment/anl_llama/env/vars.sh
source /home/jmitche1/70Bccl/libraries.performance.communication.oneccl/build/_install/env/vars.sh

#source activate anl_llma-70b
conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-70b

cd /lus/gila/projects/Aurora_deployment/anl_llama/70B/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/inference
#mpirun -np 4 ./run_script.sh  python -u run_generation_with_deepspeed.py -m /home/jmitche1/huggingface/llama2 --benchmark --input-tokens=32 --max-new-tokens=32 |& tee llma-70-1.log
mpirun -np 4 ./run_script.sh  python -u run_generation_with_deepspeed.py -m /lus/gila/projects/Aurora_deployment/anl_llama/model_weights/huggingface/llama2 --benchmark --input-tokens=1024 --max-new-tokens=128 |& tee llma-70.log
```

<a name="Inference_with_Parsl"></a>
## Inference with Parsl - 13B and 70B Llama Inference
We can use parsl to run the inference using the python script 

1. Clone this repository. Ensure you have your the public key added to user token setup correctly in Github to be able to clone repository using ssh
```bash
 git clone https://github.com/atanikan/LLM-service-api.git
```

2. Using the pre-built conda environments
   On the login node:

   *For 13B* 
   ```bash
   source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate
   conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-13b
   ```
   *For 70B* 
   ```bash
   source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate
   conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-70b
   ```


3. Now to run the parsl script for 13B, head to `LLM-service-api/parsl_service_13b` and run the following on a login node
```bash
cd LLM-service-api/parsl_service_13b
python parsl_service.py /lus/gila/projects/Aurora_deployment/anl_llama/13B/intel-extension-for-pytorch/examples/gpu/inference/python/llm/text-generation/run_llama.py --device xpu --model-dir "/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/llma_models/llma-2-convert13B" --dtype float16 --ipex --greedy
```
4. You can do the same for 70B
```bash
cd LLM-service-api/parsl_service_70b
python parsl_service.py /lus/gila/projects/Aurora_deployment/anl_llama/70B/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/inference/run_generation_with_deepspeed.py -m "/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/huggingface/llama2 --benchmark --input-tokens=1024 --max-new-tokens=128" --dtype float16 --ipex --greedy
```

:bulb: **Note:** The [config files](./parsl_service_13b/parsl_config.py) set the necessary configuration for codes to run on Sunspot on 12 tiles per node (6 GPUs)

**Note** If you don't have parsl in your custom environment, you can install parsl as follows.

```bash
conda activate ~/environments/anl_llma-13b
pip install parsl
conda deactivate
conda activate ~/environments/anl_llma-70b
pip install parsl
```

<a name="Restful_Inference_Service"></a>
## Adding a RESTAPI calls to the model to achieve Inference as a service (Work In Progress)
:bulb: **Note:** You can directly use the conda environment `/lus/gila/projects/Aurora_deployment/conda_env_llm` and skip conda setup step 1  below. Just run `conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-13b` OR `conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-70b` 

1. To use the API to make rest api calls to parsl. Activate any existing conda environment
```bash
conda activate ~/environments/anl_llma-13b
pip install fastapi[all]
```

2. SSH tunnel to the login node of sunspot and `localhost:8000/docs` will help you interact with parsl

```bash
ssh -L 8000:127.0.0.1:8000 -J username@bastion.alcf.anl.gov username@sunspot
```

3. Run server
```
cd /LLM-service-api
uvicorn LLM_service_api:app --reload
```

<a name="FuncX"></a>
## Deploying the model through Globus Compute Endpoint to achieve Inference as a service (Work In Progress)

This approach will deploy infrence tasks to a Globus Compute endpoint.

### Set-up Environment

To use the pre-built conda environments on the login node, do this:

   *For 13B* 
   ```bash
   source /soft/datascience/conda-2023-01-31/miniconda3/bin/activate
   conda activate /lus/gila/projects/Aurora_deployment/conda_env_llm/anl_llma-13b
   ```

To build your own environment (instead of using the pre-built one) follow these steps:

1. In a conda or virtual environment on Sunspot, install the necessary packages to run Globus Compute:
```
pip install globus-compute-sdk globus-compute-endpoint
```

2. It is necessary to hack your installation of `globus-compute-endpoint` to tell the endpoint process to communicate through a local port of your choosing, rather than attempting a TCP connection itself.

In your environment, look for the `endpoint.py` file in your installation's `globus_compute_endpoint/endpoint` directory.  For example, in a local `miniconda` installation that has an environment called `anl_llma-13b` it would be here:
```
~/miniconda3/envs/anl_llma-13b/lib/python3.9/site-packages/globus_compute_endpoint/endpoint/endpoint.py
```

In this file, look for the routine `start_endpoint`.  Within this routine look for the creation of an object called `reg_info`.  This object contains the endpoint registration info that needs to be edited.  Directly below `reg_info` paste the following code:
```
		# This reg_info object should be around line 357
       		reg_info = fx_client.register_endpoint(
                    endpoint_dir.name,
                    endpoint_uuid,
                    metadata=Endpoint.get_metadata(endpoint_config),
                    multi_tenant=False,
                    display_name=endpoint_config.display_name,
                )

		# Add this code
		log.info(f"reg_info: {reg_info}")
                
                def _use_my_stunnel(q_info):
                    q_info["connection_url"] = (
                        q_info["connection_url"]
                        .replace("@amqps.funcx.org", "@localhost:50000")
                        .replace("amqps://", "amqp://")
                    )

		_use_my_stunnel(reg_info["task_queue_info"])
                _use_my_stunnel(reg_info["result_queue_info"])

		log.info(f"reg_info: {reg_info}")
		# End of added code block

```

Note the use of `localhost:50000`; this is the local port on the sunspot login node that the endpoint process will look to connect through.  That port number will be necessary in later steps of making the tunnel.  If port `50000` is unavailable for some reason, another similarly high numbered port should work, but you must replace `50000` with that port number everywhere in these instructions.

### Configure Endpoint

On a Sunspot login node do the following:

1. Clone this repo.

2. On Sunspot, at the path `~/.globus_compute/<ENDPOINT_NAME>` there is a file called `config.yaml`.  We want to replace that file with a config file similar to the sample file [config.yaml](funcx_servic_13b/config.yaml).  

In the sample `config.yaml` file provided, replace your `account` and `worker_init` script path and copy it to `~/.globus_compute/<ENDPOINT_NAME>`.  The `worker_init` command should point to the script [worker_init.sh](funcx_service_13b/worker_init.sh).

### Set-up tunnel

1. On your local machine, install [`stunnel`](https://www.stunnel.org).  If you have `homebrew` on your local machine, you can use it to install:
```
brew update
brew install stunnel
```

2. Also on your local machine, create an `stunnel` configuration file.  Save it to a path of your choice `/path/to/stunnel.conf`.  The file will look like this:
```
pid = /tmp/stunnel.pid # you can set this to any path of your choice
output = /tmp/stunnel.log # you can set this to any path of your choice
foreground = no # if you want the connection to run in your shell, set this to yes

debug = 7

[amqp]
client = yes
accept = 127.0.0.1:15672 # this port can be anything of your choice
connect = amqps.funcx.org:5671 # this address and port must remain fixed

```

3. Start stunnel connection:
```
stunnel /path/to/stunnel.conf
```
If you have set foreground to `no` in `stunnel.conf` it will run in a daemon process in the background; if you set it to `yes`, it will run in the shell.

4. Now create a tunnel from your local machine to a sunspot login node.
```
ssh -R 15672:localhost:15672 bastion.alcf.anl.gov
ssh -R 50000:localhost:15672 sunspot.alcf.anl.gov
```

5. You should be in a shell on a sunspot login node that has a connection to the Globus Compute `amqps` service.  Now start the endpoint process:
```
globus-compute-endpoint start <ENDPOINT_NAME>
```

You will also need the endpoint UUID for setting up your compute tasks.  You can find it with the command:
```
globus-compute-endpoint	list
```
Copy and save the UUID of the endpoint.  

Note, anytime you want to make changes to the endpoint config, you must restart the endpoint:
```
globus-compute-endpoint restart <ENDPOINT_NAME>
```

### Run the 13b model with funcX

Return to a shell on your local machine (or any machine with a `http_proxy`, you can also do this part from a Sunspot node too).

1. Clone this repo.

2. In a local conda or python virtual environment, install the globus module:
```
pip install globus-compute-sdk
```

2. Go to `LLM-service-api/funcx_service_13b`.

3. Register the function with the globus service.  You will be asked to validate your credentials with the globus service:
```
python register_function.py
```
This routine will print the function id.  Copy and save this.

4. Edit `run_batch.py` in this directory.  Replace the `sunspot_endpoint` and `function_id` with the UUIDs from previous steps.

5. Run `run_batch.py`:
```
python run_batch.py /lus/gila/projects/Aurora_deployment/anl_llama/13B/intel-extension-for-pytorch/examples/gpu/inference/python/llm/text-generation/run_llama.py --device xpu --model-dir /lus/gila/projects/Aurora_deployment/anl_llama/model_weights/llma_models/llma-2-convert13B --dtype float16 --ipex --greedy
```

6. Inspect outputs.  The function outputs will be returned locally to where you are running `run_batch.py` in a directory `outputs/<RANDOM_UUID>/<i>/`.
```
cat ./outputs/<RANDOM_UUID>/*/llama.stdout
```

### Clean-up

- Endpoint: Stop the endpoint with 
```
globus-compute-endpoint stop <ENDPOINT_NAME>
```

- Tunnels: To close the ssh tunnels from your local machine to Sunspot, exit out of the ssh sessions you have opened in your shell.

    On your local machine, stop the stunnel connection with:
```
kill 'cat /tmp/stunnel.pid'
```

