#!/usr/bin/env python3
"""
Neuron Activation Analyzer.
Tracks binary activation states of FFN intermediate neurons.
"""

import torch
import jsonlines
import gc
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor


def get_args():
    parser = argparse.ArgumentParser(description='Qwen2-VL neuron activation analysis')

    parser.add_argument("--in_file_path", type=str, required=True,
                        help="Input JSONL file path")
    parser.add_argument("--visualize_path", type=str, required=True,
                        help="Output visualization path (.pdf or .png)")
    parser.add_argument("--pretrained_model_path", type=str,
                        default="Qwen/Qwen2-VL-7B-Instruct", help="Model path")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Image directory")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum samples to analyze")

    return parser.parse_args()


class Qwen2VLActivationCollector:
    """Collects FFN neuron activations for Qwen2-VL."""

    def __init__(self, model, processor, num_layers, num_neurons, device):
        self.model = model
        self.processor = processor
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.device = device
        self.reset()

    def reset(self):
        """Reset activation statistics."""
        self.layer_flag = 0
        self.save_flag = False
        self.data_idx = 0

        self.activation_matrix_total = [None] * self.num_layers
        self.activation_matrix_common = [None] * self.num_layers
        self.activation_matrix = [None] * self.num_layers
        self.max_activate = [None] * self.num_layers
        self.min_activate = [None] * self.num_layers
        self.avg_activate = [None] * self.num_layers

    def mlp_forward_hook(self, layer_idx):
        """Create MLP forward hook for activation collection.

        Qwen2-VL MLP structure:
        - gate_proj: Linear(hidden_size -> intermediate_size)
        - up_proj: Linear(hidden_size -> intermediate_size)
        - down_proj: Linear(intermediate_size -> hidden_size)
        - act_fn: SiLU activation

        Formula: FFN(h) = W_down(SiLU(W_gate * h) * W_up * h)
        """
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            layer = self.model.model.language_model.layers[layer_idx]
        else:
            layer = self.model.model.layers[layer_idx]

        original_mlp_forward = layer.mlp.forward

        def hooked_forward(x):
            if self.save_flag:
                gate_output = layer.mlp.gate_proj(x)
                activations = layer.mlp.act_fn(gate_output)

                # Sum activations from generation start
                actmean = (activations[:, self.data_idx:, :]).sum(dim=1, keepdim=True)
                act_bin = (actmean > 0).squeeze().squeeze()

                if self.activation_matrix_total[layer_idx] is None:
                    self.activation_matrix_total[layer_idx] = act_bin.clone()
                    self.activation_matrix_common[layer_idx] = act_bin.clone()
                    self.activation_matrix[layer_idx] = act_bin.int().clone()
                    self.max_activate[layer_idx] = act_bin.sum()
                    self.min_activate[layer_idx] = act_bin.sum()
                    self.avg_activate[layer_idx] = act_bin.sum()
                else:
                    self.activation_matrix_total[layer_idx] |= act_bin
                    self.activation_matrix_common[layer_idx] &= act_bin
                    self.activation_matrix[layer_idx] += act_bin.int()
                    self.max_activate[layer_idx] = max(act_bin.sum(), self.max_activate[layer_idx])
                    self.min_activate[layer_idx] = min(act_bin.sum(), self.min_activate[layer_idx])
                    self.avg_activate[layer_idx] += act_bin.sum()

                up_output = layer.mlp.up_proj(x)
                down_proj = layer.mlp.down_proj(activations * up_output)
                self.layer_flag = (self.layer_flag + 1) % self.num_layers
            else:
                down_proj = original_mlp_forward(x)

            return down_proj

        return hooked_forward

    def attach_hooks(self):
        """Attach hooks to all MLP layers."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'language_model'):
            layers = self.model.model.language_model.layers
        else:
            layers = self.model.model.layers

        for layer_idx in range(self.num_layers):
            layers[layer_idx].mlp.forward = self.mlp_forward_hook(layer_idx)

    def find_output_start_idx(self, input_ids):
        """Find start index of assistant output tokens."""
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

    def evaluate_per_example(self, in_file_path, image_dir, max_samples=None):
        """Evaluate neuron activations per sample."""
        self.attach_hooks()
        res_data = []
        cnt = 0

        with jsonlines.open(in_file_path) as reader:
            all_datas = list(reader)

            if max_samples is not None:
                all_datas = all_datas[:max_samples]

            for data in tqdm(all_datas, desc='Analyzing'):
                cnt += 1

                if data.get('is_faithful', 0) == -1:
                    continue

                if 'image_path' in data:
                    image_path = data['image_path']
                else:
                    image_path = os.path.join(image_dir, data['image'])

                question = data.get('question_same', data.get('question', 'Are the two pictures the same?'))

                if not os.path.exists(image_path):
                    print(f"Image not found: {image_path}")
                    continue

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": question}
                    ]
                }]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                try:
                    from qwen_vl_utils import process_vision_info
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text], images=image_inputs, videos=video_inputs,
                        padding=True, return_tensors="pt"
                    )
                except ImportError:
                    inputs = self.processor(
                        text=[text], images=[image_path],
                        padding=True, return_tensors="pt"
                    )

                inputs = inputs.to(self.device)
                self.data_idx = self.find_output_start_idx(inputs['input_ids'])
                self.reset()

                self.save_flag = True
                with torch.inference_mode():
                    self.model(**inputs)

                if cnt % 50 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                # Compute average activations
                avg_activations_per_layer = [
                    (m.item() / self.num_neurons) if m is not None else 0.0
                    for m in self.avg_activate
                ]

                data['activations_avg'] = avg_activations_per_layer
                res_data.append(data)

        return res_data


def list_add(list1, list2):
    """Element-wise list addition."""
    return [x + y for x, y in zip(list1, list2)]


def draw_pic(all_datas, num_layers, num_neurons, save_path):
    """Visualize neuron activation differences."""
    right_list = [0] * num_layers
    wrong_list = [0] * num_layers
    right_len = 0
    wrong_len = 0

    for data in all_datas:
        cur_list = data['activations_avg']

        if data.get('is_faithful', 1) == 0:
            wrong_len += 1
            wrong_list = list_add(cur_list, wrong_list)
        else:
            right_len += 1
            right_list = list_add(cur_list, right_list)

    if right_len == 0 or wrong_len == 0:
        print(f"Insufficient data: Faithful={right_len}, Unfaithful={wrong_len}")
        return

    right_list = [tmp / right_len for tmp in right_list]
    wrong_list = [tmp / wrong_len for tmp in wrong_list]

    # Difference (unfaithful - faithful) scaled by 10
    diff_list = [(w - r) * 10 for w, r in zip(wrong_list, right_list)]

    x = list(range(num_layers))

    print(f'\nActivation Statistics:')
    print(f'Unfaithful: {[f"{x:.4f}" for x in wrong_list]}')
    print(f'Faithful:   {[f"{x:.4f}" for x in right_list]}')
    print(f'Diff(x10):  {[f"{x:.4f}" for x in diff_list]}')

    plt.figure(figsize=(12, 6))
    colors = ['#ffcccc'] * num_layers
    plt.bar(x, diff_list, color=colors, width=0.6, edgecolor='#ff6666', linewidth=0.7)

    plt.xlabel("Layers", fontsize=20)
    plt.ylabel("Neuron Activation Ratio Diff (x10)", fontsize=16)
    plt.title("Qwen2-VL Neuron Activation Analysis", fontsize=22, pad=20)

    if save_path.endswith('.png'):
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    elif save_path.endswith('.pdf'):
        plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(save_path + '.png', dpi=300, bbox_inches='tight')

    print(f'\nVisualization saved: {save_path}')
    plt.close()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Qwen2-VL Neuron Activation Analysis")
    print("=" * 60)
    print(f"Input: {args.in_file_path}")
    print(f"Images: {args.image_dir}")
    print(f"Output: {args.visualize_path}")
    print(f"Max samples: {args.max_samples}")

    print(f"\n1. Loading model: {args.pretrained_model_path}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.pretrained_model_path, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="auto"
    )

    processor = AutoProcessor.from_pretrained(
        args.pretrained_model_path, trust_remote_code=True
    )

    if hasattr(model.config, 'text_config'):
        num_layers = model.config.text_config.num_hidden_layers
        num_neurons = model.config.text_config.intermediate_size
    else:
        num_layers = model.config.num_hidden_layers
        num_neurons = model.config.intermediate_size

    print(f"   Layers: {num_layers}")
    print(f"   Neurons: {num_neurons}")

    print(f"\n2. Setting up collector...")
    collector = Qwen2VLActivationCollector(model, processor, num_layers, num_neurons, device)

    print(f"\n3. Analyzing samples...")
    res_data = collector.evaluate_per_example(
        args.in_file_path, args.image_dir, max_samples=args.max_samples
    )
    print(f"Processed {len(res_data)} samples")

    print(f"\n4. Generating visualization...")
    draw_pic(res_data, num_layers, num_neurons, args.visualize_path)

    # Save data
    output_data_path = args.visualize_path.rsplit('.', 1)[0] + '_data.jsonl'
    with jsonlines.open(output_data_path, mode='w') as writer:
        for data in res_data:
            writer.write(data)
    print(f"Data saved: {output_data_path}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
