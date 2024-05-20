from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model_id = "llava-hf/llava-1.5-13b-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=True,
)
processor = AutoProcessor.from_pretrained(model_id)


def query_llava_15_13b_hf(image_paths, prompt):
    """
    Query the llava with the prompt and an image.

    Parameters:
    - image: PIL Image, the image.
    - prompt: String, the prompt.
    """
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
    template = "USER:" + "<image>\n" * len(images) + "<prompt>\nASSISTANT:"
    prompt = template.replace("<prompt>", prompt)
    inputs = processor(prompt, images, return_tensors="pt").to(device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    img_path = ["/nas64/silviase/Project/prj-blink-ja/BLINK_Benchmark/assets/test.jpeg"]
    prompts = [
        "青い浮き輪は何個ありますか？",
        "青い浮き輪は何個ありますか？ 一つ選びなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から一つ選んで答えなさい (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から一つ選んで答えなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から一つ選びなさい (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から一つ選びなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から答えなさい (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から答えなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から選びなさい (A) 0 (B) 3 (C) 2 (D) 1",
        "青い浮き輪は何個ありますか？ 次の選択肢から選びなさい。 (A) 0 (B) 3 (C) 2 (D) 1",
    ]
    
    for prompt in prompts:
        print(query_llava_15_13b_hf(img_path, prompt))