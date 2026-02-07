#!/usr/bin/env python3
"""
Utility functions for model loading and common operations.
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def load_qwen2_vl_model(model_path="Qwen/Qwen2-VL-7B-Instruct", dtype=torch.bfloat16, device_map="auto"):
    """Load Qwen2-VL model with specified parameters."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def get_model_config(model):
    """Get model configuration information."""
    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
        hidden_size = model.config.text_config.hidden_size
        intermediate_size = model.config.text_config.intermediate_size
    else:
        num_layers = model.config.num_hidden_layers
        hidden_size = model.config.hidden_size
        intermediate_size = model.config.intermediate_size
    
    return {
        'num_layers': num_layers,
        'hidden_size': hidden_size,
        'intermediate_size': intermediate_size
    }


def prepare_input(processor, image_path, question):
    """Prepare input for model inference."""
    from qwen_vl_utils import process_vision_info
    from PIL import Image

    image = Image.open(image_path)
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    
    return inputs


def find_output_start_idx(input_ids, assistant_token_ids=[151644, 77091, 198]):
    """Find the start index of assistant output."""
    if isinstance(input_ids, torch.Tensor):
        token_ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
    else:
        token_ids = input_ids

    start_indices = [
        i for i in range(len(token_ids) - len(assistant_token_ids) + 1)
        if token_ids[i:i + len(assistant_token_ids)] == assistant_token_ids
    ]

    if start_indices:
        return start_indices[-1] + len(assistant_token_ids)
    else:
        return len(token_ids) // 2
