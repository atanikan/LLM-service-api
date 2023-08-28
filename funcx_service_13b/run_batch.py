import globus_compute_sdk
import os
import sys
import uuid

BATCH_SIZE=12

# Replace these IDs with your endpoint & function!
function_id = '8f28d7ed-d49b-4d24-ada2-aca7a2ad6fc2'
sunspot_endpoint = '7bf0f111-8636-49d1-9af8-c3d76de56286'

gce = globus_compute_sdk.Executor(endpoint_id=sunspot_endpoint)

def run_code(command,output_dir):
    tasks = []
    for i in range(BATCH_SIZE):
        tasks.append(gce.submit_to_registered_function(args=[command], function_id=function_id))        
    for i,t in enumerate(tasks):
        ret_code,stdout,stderr = t.result()
        task_output_dir = os.path.join(output_dir,str(i))
        os.makedirs(task_output_dir)
        if ret_code != 0:
            print(f"Task failed! {t.result()}")
        with open(os.path.join(task_output_dir,"llama.stdout"),"w") as f:
            f.write(stdout)
        with open(os.path.join(task_output_dir,"llama.stderr"),"w") as f:
            f.write(stderr)
    print("Call for model 13B completed")


if __name__ == "__main__":

    #command = 'python /lus/gila/projects/Aurora_deployment/anl_llama/13B/intel-extension-for-pytorch/examples/gpu/inference/python/llm/text-generation/run_llama.py --device xpu --model-dir /lus/gila/projects/Aurora_deployment/anl_llama/model_weights/llma_models/llma-2-convert13B --dtype float16 --ipex --greedy'
    command = " ".join(["python"]+sys.argv[1:])
    print(command)
    output_dir = os.path.join("./output",str(uuid.uuid4()))
    run_code(command,output_dir)
