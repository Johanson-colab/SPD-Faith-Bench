#!/usr/bin/env python3
"""
Quick demo script for TAM visualization.
"""

import os
import json
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from mechanism_analysis.tam import TAM
from tqdm import tqdm


def tam_demo(image_path, prompt_text, save_dir='vis_results'):
    """Demo function for TAM visualization."""
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    # Prepare input
    image = Image.open(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    # Generate with hidden states
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            pad_token_id=processor.tokenizer.eos_token_id
        )

    generated_ids = outputs.sequences

    # Compute logits from hidden states
    logits = [model.lm_head(feats[-1]) for feats in outputs.hidden_states]

    # Define special token IDs for TAM
    special_ids = {
        'img_id': [151652, 151653],
        'prompt_id': [151653, [151645, 198, 151644, 77091]],
        'answer_id': [[198, 151644, 77091, 198], -1]
    }

    # Get vision shape and inputs
    vision_shape = (inputs['image_grid_thw'][0, 1] // 2, inputs['image_grid_thw'][0, 2] // 2)
    vis_inputs = image_inputs

    # Generate TAM for each generation step
    raw_map_records = []
    for i in range(len(logits)):
        img_map = TAM(
            generated_ids[0].cpu().tolist(),
            vision_shape,
            logits,
            special_ids,
            vis_inputs,
            processor,
            os.path.join(save_dir, f'{i}.jpg'),
            i,
            raw_map_records,
            False
        )

    print(f"TAM visualizations saved to: {save_dir}")


if __name__ == "__main__":
    # Example usage
    img = "/path/to/your/image.jpg"  # Replace with actual image path
    prompt = "Are the two pictures the same?"
    
    # Create a dummy image if path doesn't exist for demo
    if not os.path.exists(img):
        # Create a simple test image
        test_img = Image.new('RGB', (512, 512), color='red')
        img = './test_image.jpg'
        test_img.save(img)
        print(f"Created test image: {img}")

    tam_demo(img, prompt, save_dir='demo_results')
    print("Demo completed!")
