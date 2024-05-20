from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image

device = "cuda"
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-34b-hf", 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    load_in_4bit=True,
) 
model.to(device)

def query_llava_16_34b_hf(image_paths, prompt):
    """
    Query the llava with the prompt and an image.

    Parameters:
    - image: PIL Image, the image.
    - prompt: String, the prompt.
    """
    images = concat_images_horizontally_with_margin(image_paths)
    template = "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\n<your_text_prompt_here><|im_end|><|im_start|>assistant\n"
    prompt = template.replace("<your_text_prompt_here>", prompt)
    inputs = processor(prompt, images, return_tensors="pt").to(device, torch.float16)
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
    return generated_text

def concat_images_horizontally_with_margin(image_paths, margin=10):
    """
    Concatenates images horizontally with a specified margin between images,
    padding with black if heights are not the same, and saves the result to a file.

    Parameters:
    - image_filenames: List of strings, where each string is the filepath to an image.
    - margin: Integer, the width of the black margin to insert between images.

    Returns:
    - new_image: PIL Image, the concatenated image.
    """
    images = [Image.open(filename) for filename in image_paths]
    max_height = max(image.height for image in images)
    total_width = sum(image.width for image in images) + margin * (len(images) - 1)
    # Create a new image with a black background
    new_image = Image.new('RGB', (total_width, max_height), (0, 0, 0))
    
    x_offset = 0
    for image in images:
        # Calculate padding to center the image vertically
        y_offset = (max_height - image.height) // 2
        new_image.paste(image, (x_offset, y_offset))
        x_offset += image.width + margin  # Add margin after each image except the last one

    return new_image

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
        print(query_llava_16_34b_hf(img_path, prompt))