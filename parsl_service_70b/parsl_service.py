import parsl
from parsl import python_app, bash_app
import glob
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from parsl_config import sunspot_config


BATCH_SIZE=1

@bash_app
def call_model_70b (command, stdout='model70b.stdout', stderr='model70b.stderr'):
    return f"echo 'Running on tile with affinity ZE_AFFINITY_MASK={os.getenv('ZE_AFFINITY_MASK')}';\
     mpirun -np 1 {os.path.dirname(command[0])}/run_script.sh python {command[0]} {' '.join(command[1:])}"

def run_code(command,output_dir):
    tasks = []
    for i in range(BATCH_SIZE):
        tasks.append(call_model_70b(command, stdout=output_dir+f"{i}/job.out", 
                                stderr=output_dir+f"{i}/job.stderr"))        
    for t in tasks:
        t.result()
    print("Call for model 70B completed")

def fetch_output(output_dir):
    for i in range(BATCH_SIZE):
        # Open the file in read mode
        with open(output_dir+f"{i}/job.out", 'r') as file:
            # Read the file content
            content = file.read()
            # Print the content
            print(content)

def fetch_errors(output_dir):
    for i in range(BATCH_SIZE):
        # Open the file in read mode
        with open(output_dir+f"{i}/job.stderr", 'r') as file:
            # Read the file content
            content = file.read()
            # Print the content
            print(content)

def main(command):
    parsl.load(sunspot_config)
    batch_iter = max([int(p.split("/")[-1]) for p in glob.glob(f"{os.path.dirname(os.path.abspath(__file__))}/runinfo/*")])
    output_dir = f"{os.path.dirname(os.path.abspath(__file__))}/model_70B_{batch_iter}/" 
    print(">",command,output_dir)   
    run_code(command, output_dir)
    fetch_output(output_dir)
    #fetch_errors(output_dir)

if __name__ == "__main__":
    #command = ['~/anl_llama/70B/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/inference/run_generation_with_deepspeed.py', '-m', '/home/jmitche1/huggingface/llama2', '--benchmark', '--input-tokens=1024', '--max-new-tokens=128']
    main(sys.argv[1:])
    