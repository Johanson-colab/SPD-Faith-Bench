#!/usr/bin/env python3
"""
MHA/FFN Layer Change Analyzer.
Analyzes representation changes in Transformer MHA and FFN sublayers.
"""

import torch
import jsonlines
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


class LayerChangeAnalyzer:
    """Analyzes MHA/FFN sublayer representation changes."""

    def __init__(self, model, processor, num_layers, device):
        self.model = model
        self.processor = processor
        self.num_layers = num_layers
        self.device = device
        self.current_sample_states = {}
        self.hooks = []

    def reset_sample_states(self):
        """Reset states for new sample."""
        self.current_sample_states = {i: {} for i in range(self.num_layers)}

    def attach_hooks(self):
        """Attach forward hooks to capture hidden states."""
        layers = self.model.model.layers

        for layer_idx in range(self.num_layers):
            layer = layers[layer_idx]

            def make_input_hook(idx):
                def hook(module, input):
                    self.current_sample_states[idx]['xl'] = input[0].detach().clone()
                return hook

            def make_attn_hook(idx):
                def hook(module, input, output):
                    attn_output = output[0]
                    residual = self.current_sample_states[idx]['xl']
                    xl_half = residual + attn_output
                    self.current_sample_states[idx]['xl_half'] = xl_half.detach().clone()
                return hook

            def make_output_hook(idx):
                def hook(module, input, output):
                    self.current_sample_states[idx]['xl_plus1'] = output[0].detach().clone()
                return hook

            h1 = layer.register_forward_pre_hook(make_input_hook(layer_idx))
            h2 = layer.self_attn.register_forward_hook(make_attn_hook(layer_idx))
            h3 = layer.register_forward_hook(make_output_hook(layer_idx))

            self.hooks.extend([h1, h2, h3])

    def compute_l2_changes(self, data_idx=0):
        """Compute L2 norm changes for MHA and FFN sublayers."""
        results = {}

        for layer_idx in range(self.num_layers):
            states = self.current_sample_states[layer_idx]

            if not all(k in states for k in ['xl', 'xl_half', 'xl_plus1']):
                continue

            xl = states['xl']
            xl_half = states['xl_half']
            xl_plus1 = states['xl_plus1']

            if xl.numel() == 0:
                continue

            def token_mean_norm(x):
                return torch.norm(x, dim=-1).mean()

            xl_norm = token_mean_norm(xl)
            xl_half_norm = token_mean_norm(xl_half)

            # L2 changes
            mha_change = token_mean_norm(xl_half - xl)
            ffn_change = token_mean_norm(xl_plus1 - xl_half)

            mha_relative = mha_change / (xl_norm + 1e-6)
            ffn_relative = ffn_change / (xl_half_norm + 1e-6)

            # Cosine similarity
            def token_mean_cosine(x1, x2):
                x1_norm = x1 / (torch.norm(x1, dim=-1, keepdim=True) + 1e-6)
                x2_norm = x2 / (torch.norm(x2, dim=-1, keepdim=True) + 1e-6)
                cosine_sim = (x1_norm * x2_norm).sum(dim=-1)
                return cosine_sim.mean()

            mha_cosine = token_mean_cosine(xl, xl_half)
            ffn_cosine = token_mean_cosine(xl_half, xl_plus1)

            # KL divergence between MHA and FFN distributions
            import torch.nn.functional as F
            delta_attn = xl_half - xl
            delta_ffn = xl_plus1 - xl_half

            B, T, D = delta_attn.shape
            delta_attn_flat = delta_attn.reshape(-1, D)
            delta_ffn_flat = delta_ffn.reshape(-1, D)

            p_attn = F.softmax(delta_attn_flat, dim=-1)
            p_ffn = F.softmax(delta_ffn_flat, dim=-1)

            kl_div = F.kl_div(p_ffn.log(), p_attn, reduction='batchmean')

            results[layer_idx] = {
                'mha_change': mha_change.item(),
                'ffn_change': ffn_change.item(),
                'mha_relative': mha_relative.item(),
                'ffn_relative': ffn_relative.item(),
                'mha_cosine': mha_cosine.item(),
                'ffn_cosine': ffn_cosine.item(),
                'kl_attn_ffn': kl_div.item(),
            }

        return results

    def find_output_start_idx(self, input_ids):
        """Find the start index of assistant output."""
        sublist = [151644, 77091, 198]

        if isinstance(input_ids, torch.Tensor):
            token_ids = input_ids[0].tolist() if input_ids.dim() > 1 else input_ids.tolist()
        else:
            token_ids = input_ids

        start_indices = [
            i for i in range(len(token_ids) - len(sublist) + 1)
            if token_ids[i:i + len(sublist)] == sublist
        ]

        if start_indices:
            return start_indices[-1] + len(sublist)
        else:
            return len(token_ids) // 2


def analyze_samples(args):
    """Main analysis function."""
    print("=" * 80)
    print("Transformer Layer Change Analysis")
    print("=" * 80)
    print(f"Input: {args.input_file}")
    print(f"Image dir: {args.image_dir}")
    print(f"Max samples: {args.max_samples}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n1. Loading model: {args.model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
    else:
        num_layers = model.config.num_hidden_layers

    print(f"   Layers: {num_layers}")

    print(f"\n2. Setting up analyzer...")
    analyzer = LayerChangeAnalyzer(model, processor, num_layers, device)
    analyzer.attach_hooks()

    # Load samples
    samples = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if idx >= args.max_samples:
                break
            samples.append(json.loads(line.strip()))

    print(f"\n3. Loaded {len(samples)} samples")

    all_results = []

    print(f"\n4. Analyzing...")
    for sample in tqdm(samples, desc="Analyzing"):
        if sample.get('is_faithful', 0) == -1:
            continue

        if 'image_path' in sample:
            image_path = sample['image_path']
        else:
            image_path = os.path.join(args.image_dir, sample.get('image', ''))

        if not os.path.exists(image_path):
            continue

        question = sample.get('question_same', sample.get('question_open', 'Are the two pictures the same?'))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question}
            ]
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)

        analyzer.reset_sample_states()

        with torch.no_grad():
            model(**inputs)

        data_idx = analyzer.find_output_start_idx(inputs['input_ids'])
        changes = analyzer.compute_l2_changes(data_idx)

        result = {
            'image': sample.get('image', ''),
            'is_faithful': sample.get('is_faithful', -1),
            'layer_changes': changes
        }
        all_results.append(result)

    # Save results
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n5. Results saved: {args.output_file}")

    print(f"\n6. Generating visualization...")
    visualize_results(all_results, args.output_pdf, num_layers)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def visualize_results(results, output_pdf, num_layers):
    """Visualize layer change analysis results."""
    faithful_results = [r for r in results if r['is_faithful'] == 1]
    unfaithful_results = [r for r in results if r['is_faithful'] == 0]

    print(f"\nSample counts: Faithful={len(faithful_results)}, Unfaithful={len(unfaithful_results)}")

    def compute_avg(samples, num_layers):
        mha_changes = [[] for _ in range(num_layers)]
        ffn_changes = [[] for _ in range(num_layers)]
        mha_relatives = [[] for _ in range(num_layers)]
        ffn_relatives = [[] for _ in range(num_layers)]
        mha_cosines = [[] for _ in range(num_layers)]
        ffn_cosines = [[] for _ in range(num_layers)]
        kl_divs = [[] for _ in range(num_layers)]

        for sample in samples:
            for layer_idx, change in sample['layer_changes'].items():
                layer_idx = int(layer_idx)
                mha_changes[layer_idx].append(change['mha_change'])
                ffn_changes[layer_idx].append(change['ffn_change'])
                mha_relatives[layer_idx].append(change['mha_relative'])
                ffn_relatives[layer_idx].append(change['ffn_relative'])
                mha_cosines[layer_idx].append(change['mha_cosine'])
                ffn_cosines[layer_idx].append(change['ffn_cosine'])
                kl_divs[layer_idx].append(change['kl_attn_ffn'])

        return {
            'mha_change': [np.mean(m) if m else 0 for m in mha_changes],
            'ffn_change': [np.mean(f) if f else 0 for f in ffn_changes],
            'mha_relative': [np.mean(m) if m else 0 for m in mha_relatives],
            'ffn_relative': [np.mean(f) if f else 0 for f in ffn_relatives],
            'mha_cosine': [np.mean(m) if m else 0 for m in mha_cosines],
            'ffn_cosine': [np.mean(f) if f else 0 for f in ffn_cosines],
            'kl_attn_ffn': [np.mean(k) if k else 0 for k in kl_divs],
        }

    faithful_avg = compute_avg(faithful_results, num_layers) if faithful_results else None
    unfaithful_avg = compute_avg(unfaithful_results, num_layers) if unfaithful_results else None

    if not faithful_avg or not unfaithful_avg:
        print("Insufficient data for visualization")
        return

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    layers = list(range(num_layers))

    # MHA L2 change
    ax = axes[0, 0]
    ax.plot(layers, faithful_avg['mha_change'], 'o-', label='Faithful', color='#2D5A96', linewidth=2)
    ax.plot(layers, unfaithful_avg['mha_change'], '^--', label='Unfaithful', color='#e4503f', linewidth=2)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('L2 Norm Change', fontsize=14)
    ax.set_title('MHA Sublayer: Absolute Change (L2)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # FFN L2 change
    ax = axes[0, 1]
    ax.plot(layers, faithful_avg['ffn_change'], 'o-', label='Faithful', color='#2D5A96', linewidth=2)
    ax.plot(layers, unfaithful_avg['ffn_change'], '^--', label='Unfaithful', color='#e4503f', linewidth=2)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('L2 Norm Change', fontsize=14)
    ax.set_title('FFN Sublayer: Absolute Change (L2)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # MHA relative change
    ax = axes[1, 0]
    ax.plot(layers, faithful_avg['mha_relative'], 'o-', label='Faithful', color='#2D5A96', linewidth=2)
    ax.plot(layers, unfaithful_avg['mha_relative'], '^--', label='Unfaithful', color='#e4503f', linewidth=2)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Relative Change', fontsize=14)
    ax.set_title('MHA Sublayer: Relative Change', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # FFN relative change
    ax = axes[1, 1]
    ax.plot(layers, faithful_avg['ffn_relative'], 'o-', label='Faithful', color='#2D5A96', linewidth=2)
    ax.plot(layers, unfaithful_avg['ffn_relative'], '^--', label='Unfaithful', color='#e4503f', linewidth=2)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Relative Change', fontsize=14)
    ax.set_title('FFN Sublayer: Relative Change', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # MHA cosine
    ax = axes[2, 0]
    ax.plot(layers, faithful_avg['mha_cosine'], 'o-', label='Faithful', color='#2D5A96', linewidth=2)
    ax.plot(layers, unfaithful_avg['mha_cosine'], '^--', label='Unfaithful', color='#e4503f', linewidth=2)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_title('MHA Sublayer: Direction Similarity', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim([0.7, 1.0])

    # FFN cosine
    ax = axes[2, 1]
    ax.plot(layers, faithful_avg['ffn_cosine'], 'o-', label='Faithful', color='#2D5A96', linewidth=2)
    ax.plot(layers, unfaithful_avg['ffn_cosine'], '^--', label='Unfaithful', color='#e4503f', linewidth=2)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('Cosine Similarity', fontsize=14)
    ax.set_title('FFN Sublayer: Direction Similarity', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_ylim([0.7, 1.0])

    # KL divergence per layer
    ax = axes[3, 0]
    ax.plot(layers, faithful_avg['kl_attn_ffn'], 'o-', label='Faithful', color='#2D5A96', linewidth=2)
    ax.plot(layers, unfaithful_avg['kl_attn_ffn'], '^--', label='Unfaithful', color='#e4503f', linewidth=2)
    ax.set_xlabel('Layer', fontsize=14)
    ax.set_ylabel('KL Divergence', fontsize=14)
    ax.set_title('KL Divergence: softmax(MHA) || softmax(FFN)', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)

    # Average KL bar chart
    ax = axes[3, 1]
    faithful_kl_mean = np.mean(faithful_avg['kl_attn_ffn'])
    unfaithful_kl_mean = np.mean(unfaithful_avg['kl_attn_ffn'])
    bars = ax.bar(['Faithful', 'Unfaithful'], [faithful_kl_mean, unfaithful_kl_mean],
                  color=['#2D5A96', '#e4503f'], edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Average KL Divergence', fontsize=14)
    ax.set_title('Average KL Divergence Across Layers', fontsize=16)
    ax.grid(alpha=0.3, axis='y')
    for bar, value in zip(bars, [faithful_kl_mean, unfaithful_kl_mean]):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{value:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_pdf, dpi=300, bbox_inches='tight')
    print(f"Visualization saved: {output_pdf}")


def main():
    parser = argparse.ArgumentParser(description='Transformer layer change analyzer')
    parser.add_argument('--input_file', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON file')
    parser.add_argument('--output_pdf', type=str, required=True, help='Output PDF file')
    parser.add_argument('--model_path', type=str, default='Qwen/Qwen2-VL-7B-Instruct', help='Model path')
    parser.add_argument('--image_dir', type=str, required=True, help='Image directory')
    parser.add_argument('--max_samples', type=int, default=200, help='Maximum samples')

    args = parser.parse_args()
    analyze_samples(args)


if __name__ == "__main__":
    main()
