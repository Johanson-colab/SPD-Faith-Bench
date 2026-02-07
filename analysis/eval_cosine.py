#!/usr/bin/env python3
"""
Hidden State Cosine Similarity Analysis.
Analyzes contradiction faithfulness via hidden state information flow.
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


class ContradictionFaithfulnessDetector:
    """Contradiction detection via hidden state analysis."""

    QUESTION_SAME = "Are the two pictures same?"
    QUESTION_DIFF = "Are the two pictures different?"

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load vision-language model."""
        if self.model_name == "qwen":
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            model_path = "Qwen/Qwen2-VL-7B-Instruct"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_path)

        elif self.model_name == "internvl":
            from transformers import AutoModel, AutoTokenizer
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode

            model_path = "OpenGVLab/InternVL2_5-8B"
            self.model = AutoModel.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                trust_remote_code=True, device_map='auto'
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )

            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD = (0.229, 0.224, 0.225)
            self.image_transform = T.Compose([
                T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        print(f"Model loaded: {self.model_name}")

    def get_hidden_states_with_attention(self, image_path: str, question: str
                                         ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Get hidden states and attention weights."""
        if self.model_name == "qwen":
            return self._get_qwen_hidden_states(image_path, question)
        elif self.model_name == "internvl":
            return self._get_internvl_hidden_states(image_path, question)

    def _get_qwen_hidden_states(self, image_path: str, question: str):
        """Get hidden states for Qwen model."""
        from qwen_vl_utils import process_vision_info

        image = Image.open(image_path)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question}
        ]}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=512,
                output_hidden_states=True, output_attentions=True,
                return_dict_in_generate=True
            )

        generated_ids = outputs.sequences
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True
        )[0]

        # Extract hidden states and attentions
        hidden_states_list = []
        attentions_list = []

        for step_hidden, step_attn in zip(outputs.hidden_states, outputs.attentions):
            hs_stack = torch.stack([layer_hs[:, -1, :] for layer_hs in step_hidden], dim=0)
            hidden_states_list.append(hs_stack)

            attn_pooled = torch.stack([
                layer_attn[:, :, -1, :].mean(dim=2) for layer_attn in step_attn
            ], dim=0)
            attentions_list.append(attn_pooled)

        hidden_states = torch.stack(hidden_states_list, dim=1)
        attentions = torch.stack(attentions_list, dim=1).squeeze(2)

        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(2)

        return hidden_states, attentions, output_text

    def _get_internvl_hidden_states(self, image_path: str, question: str):
        """Get hidden states for InternVL model."""
        image = Image.open(image_path).convert('RGB')
        pixel_values = self.image_transform(image).unsqueeze(0).to(self.device, torch.bfloat16)

        generation_config = dict(max_new_tokens=512, do_sample=False)

        with torch.no_grad():
            response, _ = self.model.chat(
                self.tokenizer, pixel_values, question,
                generation_config, history=None, return_history=True
            )

        # Get hidden states from language model
        question_with_img = f"<image>\n{question}"
        inputs = self.tokenizer(question_with_img, return_tensors='pt', padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model.language_model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True, output_attentions=True,
                return_dict=True
            )

        hidden_states = torch.stack(outputs.hidden_states[1:], dim=0).squeeze(1)
        attentions = torch.stack(outputs.attentions, dim=0).squeeze(1)
        attentions = attentions.mean(dim=2).unsqueeze(1)

        return hidden_states, attentions, response

    def compute_cosine_similarity(self, hs1: torch.Tensor, hs2: torch.Tensor) -> List[float]:
        """Compute layer-wise cosine similarity."""
        num_layers = min(hs1.shape[0], hs2.shape[0])
        similarities = []

        for layer in range(num_layers):
            h1_pooled = hs1[layer].mean(dim=0)
            h2_pooled = hs2[layer].mean(dim=0)
            cos_sim = F.cosine_similarity(h1_pooled.unsqueeze(0), h2_pooled.unsqueeze(0))
            similarities.append(cos_sim.item())

        return similarities

    def compute_icr_score(self, hidden_states: torch.Tensor, attentions: torch.Tensor,
                          use_original: bool = False) -> List[float]:
        """Compute ICR (Information Contribution Rate) Score."""
        if use_original:
            return self._compute_icr_original(hidden_states, attentions)
        else:
            return self._compute_icr_simplified(hidden_states)

    def _compute_icr_simplified(self, hidden_states: torch.Tensor) -> List[float]:
        """Simplified ICR using L2 norm differences."""
        num_layers = hidden_states.shape[0]
        icr_scores = []

        for layer in range(1, num_layers):
            hs_diff = hidden_states[layer] - hidden_states[layer - 1]
            hs_diff_norm = torch.norm(hs_diff, dim=1).mean().item()
            icr_scores.append(hs_diff_norm)

        return icr_scores

    def _compute_icr_original(self, hidden_states: torch.Tensor, attentions: torch.Tensor,
                              top_p: float = 0.1) -> List[float]:
        """Original ICR with JS divergence."""
        num_layers, num_tokens, _ = hidden_states.shape
        icr_scores = []

        for layer in range(1, num_layers):
            layer_icr_scores = []

            for token_idx in range(num_tokens):
                if attentions.dim() == 3:
                    current_token_attn = attentions[layer - 1, token_idx, :]
                    current_token_attn = current_token_attn.mean().repeat(num_tokens)
                else:
                    current_token_attn = torch.ones(num_tokens, device=hidden_states.device) / num_tokens

                top_k = max(1, int(top_p * num_tokens))
                top_k = min(top_k, num_tokens)
                _, topk_idx = torch.topk(current_token_attn, k=top_k)

                current_token_hs = hidden_states[layer, token_idx]
                previous_token_hs = hidden_states[layer - 1, token_idx]
                current_token_hs_topk = hidden_states[layer - 1, topk_idx]

                hs_diff = current_token_hs - previous_token_hs
                w_i = torch.sum(hs_diff.unsqueeze(0) * current_token_hs_topk, dim=1) / (
                    torch.norm(current_token_hs_topk, dim=1) + 1e-8
                )

                icr_score = self._js_divergence(w_i, current_token_attn[topk_idx])
                layer_icr_scores.append(icr_score)

            icr_scores.append(np.mean(layer_icr_scores))

        return icr_scores

    def _js_divergence(self, p: torch.Tensor, q: torch.Tensor) -> float:
        """Compute JS divergence."""
        p = (p - p.mean()) / max(p.std().item(), 1e-8)
        q = (q - q.mean()) / max(q.std().item(), 1e-8)

        p = F.softmax(p, dim=0)
        q = F.softmax(q, dim=0)

        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum()
        kl_qm = (q * (q / m).log()).sum()

        return (0.5 * kl_pm + 0.5 * kl_qm).item()

    def evaluate_sample(self, image_path: str) -> Dict:
        """Evaluate a single sample."""
        print(f"\nAnalyzing: {Path(image_path).name}")

        # Get hidden states for both questions
        print("  Getting 'same' question hidden states...")
        hs_same, attn_same, output_same = self.get_hidden_states_with_attention(
            image_path, self.QUESTION_SAME
        )
        print(f"      Output: {output_same[:100]}...")

        print("  Getting 'different' question hidden states...")
        hs_diff, attn_diff, output_different = self.get_hidden_states_with_attention(
            image_path, self.QUESTION_DIFF
        )
        print(f"      Output: {output_different[:100]}...")

        # Compute metrics
        cosine_similarities = self.compute_cosine_similarity(hs_same, hs_diff)
        avg_cosine_sim = np.mean(cosine_similarities)

        icr_same = self.compute_icr_score(hs_same, attn_same, use_original=False)
        icr_diff = self.compute_icr_score(hs_diff, attn_diff, use_original=False)
        icr_divergence = np.mean([abs(a - b) for a, b in zip(icr_same, icr_diff)])

        # Compute faithfulness score
        faithfulness_score = avg_cosine_sim * (1 - min(icr_divergence, 1.0))

        result = {
            'image': Path(image_path).name,
            'output_same': output_same,
            'output_different': output_different,
            'hs_cosine_similarity': cosine_similarities,
            'icr_same': icr_same,
            'icr_diff': icr_diff,
            'icr_divergence': icr_divergence,
            'avg_cosine_similarity': avg_cosine_sim,
            'faithfulness_score': faithfulness_score
        }

        print(f"  Avg Cosine Similarity: {avg_cosine_sim:.4f}")
        print(f"  ICR Divergence: {icr_divergence:.4f}")
        print(f"  Faithfulness Score: {faithfulness_score:.4f}")

        return result

    def analyze_samples(self, comparison_result_file: str, image_dir: str,
                        output_dir: str, max_samples: int = None):
        """Analyze contradiction samples."""
        print(f"\n{'='*60}")
        print(f"Hidden State Faithfulness Analysis")
        print(f"{'='*60}")

        with open(comparison_result_file, 'r') as f:
            comparison_data = json.load(f)

        contradictory_samples = [
            r for r in comparison_data['results'] if r.get('is_contradictory', False)
        ]
        faithful_samples = [
            r for r in comparison_data['results'] if not r.get('is_contradictory', False)
        ]

        print(f"Contradictory samples: {len(contradictory_samples)}")
        print(f"Faithful samples: {len(faithful_samples)}")

        if max_samples:
            contradictory_samples = contradictory_samples[:max_samples]
            faithful_samples = faithful_samples[:max_samples]

        # Analyze contradictory samples
        contradictory_results = []
        print(f"\nAnalyzing contradictory samples ({len(contradictory_samples)})...")
        for idx, sample in enumerate(tqdm(contradictory_samples)):
            image_name = sample['image']
            pattern = os.path.join(image_dir, f"{image_name}_*.jpg")
            matching_files = glob.glob(pattern)

            if matching_files:
                result = self.evaluate_sample(matching_files[0])
                result['contradiction_type'] = sample.get('contradiction_type')
                result['sample_id'] = image_name
                contradictory_results.append(result)

        # Analyze faithful samples (subset)
        faithful_results = []
        max_faithful = min(3, len(faithful_samples))
        print(f"\nAnalyzing faithful samples ({max_faithful} for reference)...")
        for sample in tqdm(faithful_samples[:max_faithful]):
            image_name = sample['image']
            pattern = os.path.join(image_dir, f"{image_name}_*.jpg")
            matching_files = glob.glob(pattern)

            if matching_files:
                result = self.evaluate_sample(matching_files[0])
                faithful_results.append(result)

        # Compute statistics
        stats = self._compute_statistics(contradictory_results, faithful_results)

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{self.model_name}_faithfulness_analysis.json")

        with open(output_file, 'w') as f:
            json.dump({
                'model': self.model_name,
                'contradictory_samples': contradictory_results,
                'faithful_samples': faithful_results,
                'statistics': stats
            }, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved: {output_file}")
        self._visualize_results(contradictory_results, faithful_results, output_dir)

        return stats

    def _compute_statistics(self, contradictory_results: List[Dict],
                            faithful_results: List[Dict]) -> Dict:
        """Compute statistics."""
        def extract_metrics(results):
            return {
                'avg_cosine_sim': np.mean([r['avg_cosine_similarity'] for r in results]),
                'avg_icr_divergence': np.mean([r['icr_divergence'] for r in results]),
                'avg_faithfulness_score': np.mean([r['faithfulness_score'] for r in results])
            }

        contra_stats = extract_metrics(contradictory_results) if contradictory_results else {}
        faith_stats = extract_metrics(faithful_results) if faithful_results else {}

        print(f"\n{'='*60}")
        print(f"Statistics")
        print(f"{'='*60}")
        if contra_stats:
            print(f"Contradictory:")
            print(f"  Avg Cosine Sim: {contra_stats['avg_cosine_sim']:.4f}")
            print(f"  Avg ICR Divergence: {contra_stats['avg_icr_divergence']:.4f}")
            print(f"  Avg Faithfulness: {contra_stats['avg_faithfulness_score']:.4f}")
        if faith_stats:
            print(f"Faithful:")
            print(f"  Avg Cosine Sim: {faith_stats['avg_cosine_sim']:.4f}")
            print(f"  Avg ICR Divergence: {faith_stats['avg_icr_divergence']:.4f}")
            print(f"  Avg Faithfulness: {faith_stats['avg_faithfulness_score']:.4f}")

        return {'contradictory': contra_stats, 'faithful': faith_stats}

    def _visualize_results(self, contradictory_results: List[Dict],
                           faithful_results: List[Dict], output_dir: str):
        """Visualize analysis results."""
        if not contradictory_results or not faithful_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Layer-wise cosine similarity
        ax1 = axes[0, 0]
        contra_cosine = np.array([r['hs_cosine_similarity'] for r in contradictory_results])
        faith_cosine = np.array([r['hs_cosine_similarity'] for r in faithful_results])

        ax1.plot(contra_cosine.mean(axis=0), label='Contradictory', marker='o', linewidth=2)
        ax1.plot(faith_cosine.mean(axis=0), label='Faithful', marker='s', linewidth=2)
        ax1.fill_between(
            range(len(contra_cosine.mean(axis=0))),
            contra_cosine.mean(axis=0) - contra_cosine.std(axis=0),
            contra_cosine.mean(axis=0) + contra_cosine.std(axis=0),
            alpha=0.2
        )
        ax1.set_xlabel('Layer', fontsize=12)
        ax1.set_ylabel('Cosine Similarity', fontsize=12)
        ax1.set_title('Layer-wise Cosine Similarity', fontsize=14)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # ICR divergence boxplot
        ax2 = axes[0, 1]
        contra_icr = [r['icr_divergence'] for r in contradictory_results]
        faith_icr = [r['icr_divergence'] for r in faithful_results]
        ax2.boxplot([contra_icr, faith_icr], labels=['Contradictory', 'Faithful'])
        ax2.set_ylabel('ICR Divergence', fontsize=12)
        ax2.set_title('ICR Divergence Distribution', fontsize=14)
        ax2.grid(axis='y', alpha=0.3)

        # Faithfulness score histogram
        ax3 = axes[1, 0]
        ax3.hist([r['faithfulness_score'] for r in contradictory_results],
                 bins=20, alpha=0.6, label='Contradictory', color='red')
        ax3.hist([r['faithfulness_score'] for r in faithful_results],
                 bins=20, alpha=0.6, label='Faithful', color='green')
        ax3.set_xlabel('Faithfulness Score', fontsize=12)
        ax3.set_ylabel('Frequency', fontsize=12)
        ax3.set_title('Faithfulness Score Distribution', fontsize=14)
        ax3.legend()
        ax3.grid(alpha=0.3)

        # Scatter plot
        ax4 = axes[1, 1]
        ax4.scatter(
            [r['avg_cosine_similarity'] for r in contradictory_results],
            [r['icr_divergence'] for r in contradictory_results],
            c='red', alpha=0.6, label='Contradictory', s=50
        )
        ax4.scatter(
            [r['avg_cosine_similarity'] for r in faithful_results],
            [r['icr_divergence'] for r in faithful_results],
            c='green', alpha=0.6, label='Faithful', s=50
        )
        ax4.set_xlabel('Avg Cosine Similarity', fontsize=12)
        ax4.set_ylabel('ICR Divergence', fontsize=12)
        ax4.set_title('Cosine Similarity vs ICR Divergence', fontsize=14)
        ax4.legend()
        ax4.grid(alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f'{self.model_name}_faithfulness_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Hidden state faithfulness analysis')
    parser.add_argument('--model', type=str, required=True, choices=['qwen', 'internvl'],
                        help='Model to use')
    parser.add_argument('--comparison_result', type=str, required=True,
                        help='Comparison result JSON file')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Image directory')
    parser.add_argument('--output_dir', type=str, default='./eval_results_faithfulness',
                        help='Output directory')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to analyze')

    args = parser.parse_args()

    detector = ContradictionFaithfulnessDetector(args.model)
    detector.analyze_samples(
        comparison_result_file=args.comparison_result,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()
