from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

# Load the model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

def query_qwen(image_paths, prompt):
    query = tokenizer.from_list_format([
        {'image': image_path} for image_path in image_paths
    ] + [{'text': prompt}])
    response, history = model.chat(tokenizer, query=query, history=None)
    return response