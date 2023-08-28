import globus_compute_sdk
gc = globus_compute_sdk.Client()

def call_model_13b (command):
    import subprocess
    
    res = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    return res.returncode, res.stdout.decode("utf-8"), res.stderr.decode("utf-8")

model_13b = gc.register_function(call_model_13b)
print(f"Function id for model_13b is {model_13b}")
