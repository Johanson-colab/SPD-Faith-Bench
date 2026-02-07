#!/usr/bin/env python3
"""
Generation-Phase Attention Analysis.
Tracks token-wise attention allocation during model generation.
"""

import math
import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import (
    Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer,
    AutoModel, LlavaForConditionalGeneration
)
import argparse
import glob
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


def load_multi_diff_data(multi_diff_dir, max_samples=30, sample_ids=None):
    """Load multi_diff dataset samples."""
    if not os.path.exists(multi_diff_dir):
        raise FileNotFoundError(f"Directory not found: {multi_diff_dir}")

    sample_dirs = sorted([d for d in os.listdir(multi_diff_dir)
                          if os.path.isdir(os.path.join(multi_diff_dir, d))])

    samples = []
    for sample_dir in sample_dirs:
        sample_id = sample_dir

        if sample_ids is not None:
            try:
                if int(sample_id) not in sample_ids:
                    continue
            except ValueError:
                if sample_id not in [str(sid) for sid in sample_ids]:
                    continue
        else:
            if len(samples) >= max_samples:
                break

        image_path = os.path.join(multi_diff_dir, sample_dir, "merged.jpg")
        if os.path.exists(image_path):
            samples.append({
                'qid': sample_id,
                'image_path': image_path,
                'question': "Are the two pictures the same?",
                'label': ''
            })

    return samples


def analyze_generation_attention(
    model_path, image_dir=None, multi_diff_dir=None,
    max_samples=3, max_new_tokens=512, device="cuda:0",
    layer_start=0, layer_end=27, sample_ids=None, model_type="qwen"
):
    """Analyze attention patterns during generation."""

    cache_dir = os.environ.get('HF_HOME', '/tmp/huggingface')

    print("Loading model...")
    if model_type == "llava":
        processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map=device, cache_dir=cache_dir
        ).eval()
        process_vision_info = None

    elif model_type == "internvl":
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map=device,
            cache_dir=cache_dir, trust_remote_code=True
        ).eval()
        processor = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, cache_dir=cache_dir
        )
        process_vision_info = None

    else:  # qwen
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            process_vision_info = None

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation="eager", device_map=device, cache_dir=cache_dir
        ).eval()

        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True,
            padding_side='left', use_fast=True, cache_dir=cache_dir
        )

    # Load samples
    if multi_diff_dir:
        print(f"Loading from: {multi_diff_dir}")
        samples = load_multi_diff_data(multi_diff_dir, max_samples, sample_ids)
    elif image_dir:
        print(f"Loading from: {image_dir}")
        image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))[:max_samples]
        samples = [{
            'qid': os.path.basename(f).replace('.jpg', ''),
            'image_path': f,
            'question': "What's the differences between two pictures?",
            'label': ''
        } for f in image_files]
    else:
        raise ValueError("Provide either multi_diff_dir or image_dir")

    print(f"Loaded {len(samples)} samples")

    all_results = []

    for idx, sample in enumerate(tqdm(samples, desc="Analyzing")):
        sample_id = str(sample['qid'])
        image_path = sample['image_path']
        question = sample['question']

        print(f"\n{'='*80}")
        print(f"Sample {idx + 1}/{len(samples)} (ID: {sample_id})")
        print(f"{'='*80}")

        image = Image.open(image_path)

        if model_type == "llava":
            image_pil = Image.open(image_path).convert('RGB')
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = processor(text=prompt, images=image_pil, return_tensors="pt", padding=True).to(device)

            input_ids = inputs['input_ids'][0].tolist()
            input_len = len(input_ids)

            IMAGE_TOKEN_INDEX = 32000
            NUM_IMG_TOKENS = 576

            if IMAGE_TOKEN_INDEX in input_ids:
                pos = input_ids.index(IMAGE_TOKEN_INDEX)
                sys_len = pos
                img_len = NUM_IMG_TOKENS
                pos_end = pos + NUM_IMG_TOKENS
            else:
                sys_len, img_len, pos_end = 0, 0, 0

        elif model_type == "internvl":
            print("InternVL uses model.chat() API, attention analysis not supported")
            continue

        else:  # qwen
            messages_query = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }]

            image_inputs, _ = process_vision_info(messages_query)
            text_query = processor.apply_chat_template(messages_query, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text_query], images=image_inputs, padding=True, return_tensors="pt").to(device)

            input_ids = inputs['input_ids'][0].tolist()

            vision_start_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            vision_end_token_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

            try:
                pos = input_ids.index(vision_start_token_id) + 1
                pos_end = input_ids.index(vision_end_token_id)
            except ValueError:
                print(f"Vision tokens not found for {sample_id}")
                continue

            sys_len = pos
            img_len = pos_end - pos
            input_len = len(input_ids)

        MAX_IMG_TOKENS = 2000
        if img_len > MAX_IMG_TOKENS:
            print(f"Image tokens exceed limit ({img_len} > {MAX_IMG_TOKENS}), skipping")
            continue

        print(f"Token lengths: total={input_len}, sys={sys_len}, img={img_len}")

        # Generate with attention output
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                output_attentions=True, return_dict_in_generate=True,
                pad_token_id=processor.eos_token_id if model_type == "internvl" else processor.tokenizer.eos_token_id,
                eos_token_id=processor.eos_token_id if model_type == "internvl" else processor.tokenizer.eos_token_id,
            )

        generated_ids = outputs.sequences[0][input_len:].tolist()
        if model_type == "qwen":
            generated_tokens = [processor.tokenizer.decode([tid]) for tid in generated_ids]
        else:
            generated_tokens = [processor.decode([tid]) for tid in generated_ids]

        print(f"Generated {len(generated_tokens)} tokens")

        # Analyze attention per layer
        layers_to_analyze = list(range(layer_start, layer_end + 1))
        generation_attention = {layer_idx: [] for layer_idx in layers_to_analyze}

        for t in range(len(outputs.attentions)):
            for layer_idx in layers_to_analyze:
                attn_t = outputs.attentions[t][layer_idx]
                attn_last_query = attn_t[:, :, -1, :]
                attn_avg = attn_last_query.mean(dim=1).squeeze(0)

                # Compute attention distribution
                attn_sys = attn_avg[0:sys_len].sum().item() if sys_len > 0 else 0.0
                attn_img = attn_avg[sys_len:pos_end].sum().item()
                attn_user = attn_avg[pos_end:input_len].sum().item()

                total = attn_sys + attn_img + attn_user
                if total > 0:
                    attn_sys /= total
                    attn_img /= total
                    attn_user /= total

                generation_attention[layer_idx].append({
                    'token': generated_tokens[t] if t < len(generated_tokens) else '',
                    'attn_sys': attn_sys,
                    'attn_img': attn_img,
                    'attn_user': attn_user,
                })

        all_results.append({
            'sample_id': sample_id,
            'image_path': image_path,
            'question': question,
            'label': sample.get('label', ''),
            'generated_tokens': generated_tokens,
            'generation_attention': generation_attention,
            'input_info': {
                'sys_len': sys_len,
                'img_len': img_len,
                'input_len': input_len,
            }
        })

        del outputs, inputs
        torch.cuda.empty_cache()

    return all_results


def plot_generation_attention(results, output_dir="./generation_plots"):
    """Plot attention patterns for each sample."""
    os.makedirs(output_dir, exist_ok=True)

    if len(results) == 0:
        return

    layers = sorted(results[0]['generation_attention'].keys())

    for idx, result in enumerate(results):
        sample_id = result['sample_id']

        num_layers = len(layers)
        ncols = min(5, num_layers)
        nrows = math.ceil(num_layers / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
        if num_layers == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if num_layers > 1 else [axes]

        for i, layer_idx in enumerate(layers):
            gen_attn = result['generation_attention'][layer_idx]

            if len(gen_attn) == 0:
                continue

            token_indices = list(range(1, len(gen_attn) + 1))
            attn_sys = [item['attn_sys'] for item in gen_attn]
            attn_img = [item['attn_img'] for item in gen_attn]
            attn_user = [item['attn_user'] for item in gen_attn]

            ax = axes[i]
            ax.plot(token_indices, attn_sys, marker='o', linewidth=2, markersize=4,
                    label='System', color='#f4a460', alpha=0.8)
            ax.plot(token_indices, attn_img, marker='s', linewidth=2, markersize=4,
                    label='Visual', color='#eaa1a6', alpha=0.8)
            ax.plot(token_indices, attn_user, marker='^', linewidth=2, markersize=4,
                    label='User', color='#afd3e6', alpha=0.8)

            ax.set_xlabel('Token Index', fontsize=12)
            ax.set_ylabel('Attention', fontsize=12)
            ax.set_title(f'Layer {layer_idx}', fontsize=14)
            ax.set_xlim(0, len(token_indices) + 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)

        for j in range(num_layers, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"sample_{sample_id}_generation_attention.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Saved: {save_path}")

        # Save data
        data_path = os.path.join(output_dir, f"sample_{sample_id}_generation_data.json")
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump({
                'sample_id': sample_id,
                'question': result['question'],
                'generated_tokens': result['generated_tokens'],
                'attention_data': result['generation_attention'],
                'input_info': result['input_info']
            }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation-phase attention analysis")
    parser.add_argument("--model-path", type=str, default="Qwen/Qwen2-VL-7B-Instruct", help="Model path")
    parser.add_argument("--image-dir", type=str, default=None, help="Image directory")
    parser.add_argument("--multi-diff-dir", type=str, default=None, help="Multi-diff directory")
    parser.add_argument("--max-samples", type=int, default=30, help="Max samples")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--output-dir", type=str, default="./generation_plots", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--layer-start", type=int, default=0, help="Start layer")
    parser.add_argument("--layer-end", type=int, default=27, help="End layer")
    parser.add_argument("--sample-ids", type=str, default=None, help="Specific sample IDs (comma-separated)")
    parser.add_argument("--model-type", type=str, default="qwen", choices=["qwen", "internvl", "llava"])

    args = parser.parse_args()

    sample_ids = None
    if args.sample_ids:
        sample_ids = [int(sid.strip()) for sid in args.sample_ids.split(',')]

    if not args.multi_diff_dir and not args.image_dir:
        parser.error("Provide --multi-diff-dir or --image-dir")

    # Validate layer range
    max_layer = 31 if args.model_type == "llava" else 27
    if args.layer_start < 0 or args.layer_end > max_layer:
        raise ValueError(f"Layer range must be 0-{max_layer} for {args.model_type}")

    print("=" * 80)
    print("Generation-Phase Attention Analysis")
    print(f"Layers: {args.layer_start} - {args.layer_end}")
    print("=" * 80)

    results = analyze_generation_attention(
        model_path=args.model_path,
        image_dir=args.image_dir,
        multi_diff_dir=args.multi_diff_dir,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        sample_ids=sample_ids,
        model_type=args.model_type
    )

    print(f"\n{'='*80}")
    print("Generating visualizations")
    print("=" * 80)

    plot_generation_attention(results, args.output_dir)

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
