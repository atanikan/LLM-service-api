import parsl
from parsl import python_app, bash_app
import glob
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parsl_config import sunspot_config


BATCH_SIZE=12

@bash_app
def call_model_13b (command, stdout='model13b.stdout', stderr='model13b.stderr'):
    return f"echo 'Running on tile with affinity ZE_AFFINITY_MASK={os.getenv('ZE_AFFINITY_MASK')}'; python {command[0]} {' '.join(command[1:])}"

def run_code(command,output_dir):
    tasks = []
    for i in range(BATCH_SIZE):
        tasks.append(call_model_13b(command, stdout=output_dir+f"{i}/job.out", 
                                stderr=output_dir+f"{i}/job.stderr"))        
    for t in tasks:
        t.result()
    print("Call for model 13B completed")

def fetch_output(output_dir):
    content_str = ''
    for i in range(BATCH_SIZE):
        # Open the file in read mode
        with open(output_dir+f"{i}/job.out", 'r') as file:
            # Read the file content
            content = file.read()
            # Print the content
            print(content)
            content_str = content_str + content
    return content_str

def fetch_errors(output_dir):
    content_str = ''
    for i in range(BATCH_SIZE):
        # Open the file in read mode
        with open(output_dir+f"{i}/job.stderr", 'r') as file:
            # Read the file content
            content = file.read()
            # Print the content
            print(content)
            content_str = content_str + content
    return content_str


def main(command):
    parsl.load(sunspot_config)
    
    batch_iter = max([int(p.split("/")[-1]) for p in glob.glob(f"{os.path.dirname(os.path.abspath(__file__))}/runinfo/*")])
    output_dir = f"{os.path.dirname(os.path.abspath(__file__))}/model_13B_{batch_iter}/" 
    run_code(command, output_dir)
    return fetch_output(output_dir)
    #fetch_errors(output_dir)

#model_python_file='/lus/gila/projects/Aurora_deployment/atanikanti/rest_anl_llama/run_llama.py'
if __name__ == "__main__":
    #command = ['/lus/gila/projects/Aurora_deployment/atanikanti/rest_anl_llama/run_model_13B_test.sh', '--model-dir', '/home/jmitche1/llma_models/llma-2-convert13B', '--device', 'xpu', '--dtype', 'float16', '--max-new-tokens', '32', '--input-tokens', '32', '--num-iter', '10', '--num-warmup', '3', '--batch-size', '1', '--greedy', '--ipex']
    main(sys.argv[1:])
    