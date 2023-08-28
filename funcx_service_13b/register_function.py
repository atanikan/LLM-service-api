import globus_compute_sdk
gc = globus_compute_sdk.Client()

def call_model_13b (command, output_dir="$HOME/LLM-service-api/funcx_service_13b/output"):
    import subprocess
    import os
    
    cmd = f"echo 'Running on tile with affinity ZE_AFFINITY_MASK={os.getenv('ZE_AFFINITY_MASK')}'; python {command[0]} {' '.join(command[1:])}"                                                                                     
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, executable='/bin/bash')

    return res.returncode, res.stdout.decode("utf-8"), res.stderr.decode("utf-8")

model_13b = gc.register_function(call_model_13b)
print(f"Function id for model_13b is {model_13b}")

    
