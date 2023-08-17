# Running Llama LLM

## Running 13B Llama LLM

* Running without workflow
```

```





## Run FASTAPI server
1. To use the API to talk to parsl
* Activate conda environment
```
conda activate /lus/gila/projects/Aurora_deployment/atanikanti/environments/restapi_anl_llama_env
```
* Run server
```
cd /lus/gila/projects/Aurora_deployment/atanikanti/rest_anl_llama
uvicorn llm_api_parsl_server:app --reload
```
2. To run the code directly on a compute node
* Activate conda environment
```
conda activate /lus/gila/projects/Aurora_deployment/atanikanti/environments/restapi_anl_llama_env
```
* Run server
```
cd /lus/gila/projects/Aurora_deployment/atanikanti/rest_anl_llama
uvicorn llm_api_server:app --reload
```
## 13B Model
* To run parsl directly activate conda environment and 
```
python /lus/gila/projects/Aurora_deployment/atanikanti/rest_anl_llama/parsl_13b.py
```


