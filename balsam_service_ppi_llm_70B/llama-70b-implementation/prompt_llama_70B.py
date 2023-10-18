import sys
import os
import re
import openai
import json
import requests
import time
import pandas as pd
import subprocess

df = pd.read_csv('proteins.csv', header=None, names=['search_words'])

print(df.head())

with open("prompt_llama_70B.output.txt","w") as f:

    #openai.api_key = "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"
    #model_engine = "gpt-3.5-turbo" 
    # This specifies which GPT model to use, as there are several models available, each with different capabilities and performance characteristics.

    # open the input file and read its contents
    depth = 30
    w_list =[] # work list of proteins not prompted yet
    s_list =[] # seen list of proteins we have already prompted


    argument = sys.argv[1]

    if (not argument):
        with open('prompts.txt', 'r') as ff:
            strings = ff.readlines()
            for string in strings:
                w_list.append(string.strip())
    else:
        w_list.append(argument.strip())            

    w_list = list(set(w_list))        
    print(w_list,file=f)
    leng = len(w_list)

    while depth > 0 and leng > 0:
        depth = depth - 1
        target = w_list[0]
        s_list.append(target)
        del w_list[0]
        print("------------------------------------------",file=f)
        print("Target -->", target,file=f)
        # response = openai.ChatCompletion.create(
        #     model='gpt-3.5-turbo',
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": "We are interested in protein interactions.  Based on the documents you have been trained with, can you provide any information on which proteins might interact with " + target},
        #     ])
        script_path = "/lus/gila/projects/Aurora_deployment/atanikanti/70B-opt/intel-extension-for-transformers/examples/huggingface/pytorch/text-generation/inference/run_generation_with_deepspeed.py"
        model_run_file = "/lus/gila/projects/Aurora_deployment/anl_llama/model_weights/huggingface/llama2"
        prompt = "We are interested in protein interactions.  Based on the documents you have been trained with, can you provide any information on which proteins might interact with " + target
        command = [
            "mpirun",
            "-np", "4",
            "python",
            script_path,
            "-m", model_run_file,
            "--prompt", f"'{prompt}'",
            "--ipex",
            "--num-iter","10"        
            ]
        try:
            print("Running Llama 70B: ",command)
            response = subprocess.run(command, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("Error in subprocess")
            print(e.output)

        print("Model response>>",response.stdout)

        #replace
        result = response.stdout

        # result = ''
        # for choice in response.choices:
        #     result += choice.message.content
        #end

        print(result, file=f)
        
        # Define the string to be searched
        search_string = result

        # Define an empty list to store the matches
        matches = []

        # Loop through the words in the DataFrame
        for word in df['search_words']:
            # Check if the word is in the search string
            pattern = r"\b{}\b".format(word)
            match = re.search(pattern, search_string)
            if match:
                # If it is, append the word to the matches list
                matches.append(word)
        
            # Convert the matches list into a new DataFrame
        matches_u = list(set(matches))        
        matches_df = pd.DataFrame({'match_words': matches_u})

        # Print the new DataFrame
        matches_df = matches_df.drop_duplicates()
        for word in matches_df['match_words']:
            if target != word:
                print(target,"-->", word)
                w_list.append(word)
        w_list = list(set(w_list)) # remove duplicates before removing previously seen
        for item in s_list:
            if item in w_list:
                w_list.remove(item)
        print(w_list, file=f)
        print("-----------------------------------------------",file=f)
        print(" ",file=f)
                        
