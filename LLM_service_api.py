from fastapi import FastAPI, Query, Body, HTTPException
import subprocess
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI()

@app.post("/run-llama/13B")
async def run_llama_13(
    model_run_file: str = Body("/home/atanikanti/anl_llama/13B/intel-extension-for-pytorch/examples/gpu/inference/python/llm/text-generation/run_llama.py", description="The model file"),
    model_dir: str = Body("/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/llma_models/llma-2-convert13B", description="Model directory"),
    device: str = Query("xpu", description="Device type", enum=["cpu", "xpu", "cuda", "hpu"]),
    dtype: str = Query("float16", description="Data type", enum=["float32", "bfloat16", "float16"]),
    max_new_tokens: int = Query(32, description="Output max new tokens"),
    greedy: bool = Query(True, description="Use greedy algorithm"),
    ipex: bool = Query(True, description="Use Intel PyTorch Extension"),
    jit: bool = Query(False, description="Use JIT compilation"),
    input_tokens: str = Query("32", description="Input tokens"),
    prompt: str = Query(None, description="Prompt"),
    num_iter: int = Query(10, description="Number of iterations"),
    num_warmup: int = Query(3, description="Number of warmups"),
    batch_size: int = Query(1, description="Batch size"),
    print_memory: bool = Query(False, description="Print memory"),
    token_latency: bool = Query(False, description="Token latency"),
    disable_optimize_transformers: bool = Query(False, description="Disable optimization for transformers")
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "parsl_service_13b", "parsl_service.py")
    command = [
        "python",
        script_path,
        model_run_file,
        "--model-dir", model_dir,
        "--device", device,
        "--dtype", dtype,
        "--max-new-tokens", str(max_new_tokens),
        "--input-tokens", input_tokens,
        "--num-iter", str(num_iter),
        "--num-warmup", str(num_warmup),
        "--batch-size", str(batch_size)    
        ]

    command.append
    if greedy:
        command.append("--greedy")
    if ipex:
        command.append("--ipex")
    if jit:
        command.append("--jit")
    if prompt:
        command.extend(["--prompt", prompt])
    if print_memory:
        command.append("--print-memory")
    if token_latency:
        command.append("--token-latency")
    if disable_optimize_transformers:
        command.append("--disable_optimize_transformers")
    
    try:
        print("Running 13B: ",command)
        result = subprocess.run(command, capture_output=True, text=True)
        return {"output": result.stdout}
    except Exception as e:
        # Raising as an HTTPException will ensure that FastAPI returns
        # a proper error response to the client
        raise HTTPException(status_code=400, detail=f"Command execution failed: {e}")

@app.post("/run-llama/70B")
async def run_llama_70(
    model_run_file: str = Body("/lus/gila/projects/Aurora_deployment/anl_llama/70B/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/inference/run_generation_with_deepspeed.py", description="The model file"),
    model_dir: str = Body("/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/huggingface/llama2", description="The huggingface model id"),
    dtype: str = Query("float16", description="Data type", enum=["float32", "bfloat16", "float16"]),
    max_new_tokens: int = Query(128, description="Output max new tokens"),
    greedy: bool = Query(False, description="Use greedy algorithm"),
    ipex: bool = Query(False, description="Use Intel PyTorch Extension"),
    jit: bool = Query(False, description="Use JIT compilation"),
    input_tokens: str = Query("1024", description="Input tokens"),
    prompt: str = Query(None, description="Prompt"),
    num_iter: int = Query(1, description="Number of iterations"),
    num_warmup: int = Query(5, description="Number of warmups"),
    batch_size: int = Query(1, description="Batch size"),
    print_memory: bool = Query(False, description="Print memory"),
    token_latency: bool = Query(False, description="Token latency"),
    throughput: bool = Query(False, description="Disable throughput"),
    ki: bool = Query(False, description="ki"),
    benchmark: bool = Query(True, description="Additionally run benchmark")
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "parsl_service_70b", "parsl_service.py")
    command = [
        "python",
        script_path,
        model_run_file,
        "-m", model_dir,
        "--dtype", dtype,
        "--max-new-tokens", str(max_new_tokens),
        "--input-tokens", input_tokens,
        "--num-iter", str(num_iter),
        # "--num-warmup", str(num_warmup),
        # "--batch-size", str(batch_size),    
        ]

    if greedy:
        command.append("--greedy")
    if ipex:
        command.append("--ipex")
    if jit:
        command.append("--jit")
    if prompt:
        command.extend(["--prompt", prompt])
    if print_memory:
        command.append("--print-memory")
    if token_latency:
        command.append("--token-latency")
    if benchmark:
        command.append("--benchmark")
    if ki:
        command.append("--ki")
    if throughput:
        command.append("--throughput")    
    try:
        print("Running 70B: ",command)
        result = subprocess.run(command, capture_output=True, text=True)
        return {"output": result.stdout}
    except Exception as e:
        # Raising as an HTTPException will ensure that FastAPI returns
        # a proper error response to the client
        raise HTTPException(status_code=400, detail=f"Command execution failed: {e}")


@app.post("/run-llama/70B-opt")
async def run_llama_70_opt(
    model_run_file: str = Body("/lus/gila/projects/Aurora_deployment/70B-opt/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/inference/run_generation_with_deepspeed.py", description="The model file"),
    model_dir: str = Body("/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/huggingface/llama2", description="The huggingface model id"),
    dtype: str = Query("float16", description="Data type", enum=["float32", "bfloat16", "float16"]),
    max_new_tokens: int = Query(128, description="Output max new tokens"),
    greedy: bool = Query(False, description="Use greedy algorithm"),
    ipex: bool = Query(False, description="Use Intel PyTorch Extension"),
    jit: bool = Query(False, description="Use JIT compilation"),
    input_tokens: str = Query("1024", description="Input tokens"),
    prompt: str = Query(None, description="Prompt"),
    num_iter: int = Query(1, description="Number of iterations"),
    num_warmup: int = Query(5, description="Number of warmups"),
    batch_size: int = Query(1, description="Batch size"),
    print_memory: bool = Query(False, description="Print memory"),
    token_latency: bool = Query(False, description="Token latency"),
    throughput: bool = Query(False, description="Disable throughput"),
    ki: bool = Query(False, description="ki"),
    benchmark: bool = Query(True, description="Additionally run benchmark")
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(current_dir, "parsl_service_70b_opt", "parsl_service.py")
    command = [
        "python",
        script_path,
        model_run_file,
        "-m", model_dir,
        "--dtype", dtype,
        "--max-new-tokens", str(max_new_tokens),
        "--input-tokens", input_tokens,
        "--num-iter", str(num_iter),
        "--num-warmup", str(num_warmup),
        "--batch-size", str(batch_size),    
        ]

    if greedy:
        command.append("--greedy")
    if ipex:
        command.append("--ipex")
    if jit:
        command.append("--jit")
    if prompt:
        command.extend(["--prompt", prompt])
    if print_memory:
        command.append("--print-memory")
    if token_latency:
        command.append("--token-latency")
    if benchmark:
        command.append("--benchmark")
    if ki:
        command.append("--ki")
    if throughput:
        command.append("--throughput")    
    try:
        print("Running 70B Opt: ",command)
        result = subprocess.run(command, capture_output=True, text=True)
        return {"output": result.stdout}
    except Exception as e:
        # Raising as an HTTPException will ensure that FastAPI returns
        # a proper error response to the client
        raise HTTPException(status_code=400, detail=f"Command execution failed: {e}")

