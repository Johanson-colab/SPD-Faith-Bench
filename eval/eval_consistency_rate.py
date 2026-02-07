#!/usr/bin/env python3
"""
Consistency Rate Evaluator.
Evaluates consistency between responses to complementary questions using GPT-4o-mini.
"""

import json
import os
import glob
import base64
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional


class ConsistencyRateEvaluator:
    """Consistency Rate Evaluator using GPT-4o-mini."""

    EVALUATION_PROMPT = """You are evaluating the CONSISTENCY of a vision-language model's response.
The model was asked TWO questions about the same pair of images:
- Question A: "Are the two pictures the same?"
- Question B: "Are the two pictures different?"

**Ground Truth:**
{ground_truth}

**Model's Response to Question A:**
{response_same}

**Model's Response to Question B:**
{response_different}

**TASK: Extract and Compare Specific Difference Descriptions**

**Rules:**
1. IGNORE Yes/No judgments - focus only on SPECIFIC difference descriptions
2. Compare concrete claims between two responses
3. Classify each claim as: consistent, contradictory, or ambiguous

**Output Format (JSON):**
```json
{{
  "overall_consistency_rate": 0.85,
  "total_claims": 6,
  "consistent_claims": 5,
  "contradictory_claims": 1,
  "ambiguous_claims": 0,
  "claim_analysis": [
    {{
      "claim_id": 1,
      "claim_text_A": "The car changed from black to red",
      "claim_text_B": "Car color is different",
      "claim_type": "consistent",
      "score_contribution": 1.0
    }}
  ],
  "summary": {{
    "is_logically_consistent": true,
    "primary_issue": "One color mismatch"
  }}
}}
```

**Scoring:**
- Consistent claim: +1.0
- Contradictory claim: -1.0
- Ambiguous claim: +0.5
- consistency_rate = sum(scores) / total_claims

Now evaluate the model's consistency:"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialize evaluator with OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("Please install openai: pip install openai")

        if api_key is None:
            api_key = os.environ.get('OPENAI_API_KEY', '')
        if base_url is None:
            base_url = os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1')

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def format_ground_truth(gt_data: Dict) -> str:
        """Format ground truth for prompt."""
        modifications = gt_data.get('modifications', [])

        if not modifications:
            return "No differences (images are identical)"

        lines = [f"Total differences: {len(modifications)}\n"]
        for i, mod in enumerate(modifications, 1):
            mod_type = mod.get('type', 'unknown')
            category = mod.get('category', 'unknown')
            if mod_type == 'color':
                target_color = mod.get('target_color', 'unknown')
                lines.append(f"{i}. Type: {mod_type}, Object: {category}, Color: {target_color}")
            else:
                lines.append(f"{i}. Type: {mod_type}, Object: {category}")

        return "\n".join(lines)

    def evaluate_sample(self, img1_path: str, img2_path: str,
                        ground_truth: Dict, response_same: str,
                        response_different: str, sample_id: str = "unknown") -> Dict:
        """Evaluate consistency for a single sample."""
        gt_text = self.format_ground_truth(ground_truth)
        prompt = self.EVALUATION_PROMPT.format(
            ground_truth=gt_text,
            response_same=response_same,
            response_different=response_different
        )

        try:
            from PIL import Image
            from io import BytesIO

            def compress_image(img_path, max_size=1024):
                img = Image.open(img_path)
                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                buffer = BytesIO()
                img.save(buffer, format='JPEG', quality=85)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')

            img1_b64 = compress_image(img1_path)
            img2_b64 = compress_image(img2_path)

            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}},
                {"type": "text", "text": prompt}
            ]

            print(f"\nCalling GPT-4o-mini API (sample {sample_id})...")
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": content}],
                max_tokens=3000,
                temperature=0,
                timeout=90
            )

            evaluation_text = (response.choices[0].message.content or "").strip()
            return self.parse_evaluation_response(evaluation_text)

        except Exception as e:
            print(f"\nGPT-4o-mini API error: {e}")
            return {"overall_consistency_rate": None, "error": str(e)}

    @staticmethod
    def parse_evaluation_response(response_text: str) -> Dict:
        """Parse GPT-4o-mini evaluation response."""
        try:
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)

            if match:
                evaluation = json.loads(match.group(1))
            else:
                evaluation = json.loads(response_text)

            required_fields = ['overall_consistency_rate', 'consistent_claims', 'contradictory_claims']
            for field in required_fields:
                if field not in evaluation:
                    evaluation[field] = None if field == 'overall_consistency_rate' else 0

            return evaluation

        except json.JSONDecodeError as e:
            print(f"\nJSON parse error: {e}")
            return {"overall_consistency_rate": None, "parse_error": str(e), "raw_response": response_text}

    def evaluate_from_responses_file(self, responses_file: str, data_dir: str,
                                      output_dir: str, max_samples: Optional[int] = None):
        """Evaluate consistency from responses file."""
        print(f"\n{'='*80}")
        print(f"Processing: {responses_file}")
        print(f"{'='*80}\n")

        with open(responses_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_name = data.get('model', 'unknown')
        responses_list = data.get('responses', [])

        if max_samples:
            responses_list = responses_list[:max_samples]

        print(f"Model: {model_name}")
        print(f"Samples: {len(responses_list)}\n")

        results = []
        for item in tqdm(responses_list, desc=f"Evaluating {model_name}"):
            try:
                sample_id = item['sample_id']
                response_same = item.get('response_same', '')
                response_diff = item.get('response_different', '')
                ground_truth = item.get('ground_truth', {})

                sample_dir = os.path.join(data_dir, sample_id)
                if not os.path.exists(sample_dir):
                    print(f"\nDirectory not found: {sample_dir}, skipping")
                    continue

                original_img = os.path.join(sample_dir, f"{sample_id}_original.jpg")
                modified_imgs = glob.glob(os.path.join(sample_dir, "*.jpg"))
                modified_imgs = [f for f in modified_imgs
                                 if "original" not in os.path.basename(f)
                                 and "merged" not in os.path.basename(f)]

                if not os.path.exists(original_img) or not modified_imgs:
                    print(f"\nMissing images for {sample_id}, skipping")
                    continue

                evaluation = self.evaluate_sample(
                    original_img, modified_imgs[0],
                    ground_truth, response_same, response_diff, sample_id=sample_id
                )

                results.append({
                    "sample_id": sample_id,
                    "ground_truth": ground_truth,
                    "response_same": response_same,
                    "response_different": response_diff,
                    "consistency_evaluation": evaluation
                })

            except Exception as e:
                print(f"\nError processing: {e}")
                continue

        summary_stats = self.compute_summary_statistics(results)

        output_file = os.path.join(output_dir, f"{model_name}_consistency_results.json")
        output_data = {
            "model": model_name,
            "total_samples": len(results),
            "evaluation_method": "gpt4o_mini_consistency_check",
            "summary_statistics": summary_stats,
            "detailed_results": results
        }

        os.makedirs(output_dir, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.print_summary(model_name, summary_stats)
        print(f"\nResults saved: {output_file}\n")

        return summary_stats

    def evaluate_multiple_models(self, response_files: List[str], data_dir: str,
                                  output_dir: str, max_samples: Optional[int] = None):
        """Evaluate consistency for multiple models."""
        print(f"\n{'='*80}")
        print(f"Consistency Rate Evaluation - Powered by GPT-4o-mini")
        print(f"{'='*80}\n")

        all_model_results = {}
        for response_file in response_files:
            try:
                stats = self.evaluate_from_responses_file(
                    response_file, data_dir, output_dir, max_samples
                )

                with open(response_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                model_name = data.get('model', 'unknown')

                all_model_results[model_name] = stats

            except Exception as e:
                print(f"\nError processing {response_file}: {e}")
                continue

        comparison_file = os.path.join(output_dir, "all_models_consistency_comparison.json")
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_model_results, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*80}")
        print(f"All models comparison saved: {comparison_file}")
        print(f"{'='*80}\n")

    @staticmethod
    def compute_summary_statistics(results: List[Dict]) -> Dict:
        """Compute summary statistics."""
        valid_samples = [r for r in results
                        if r['consistency_evaluation'].get('overall_consistency_rate') is not None]

        if not valid_samples:
            return {"avg_consistency_rate": None, "total_samples": len(results), "valid_samples": 0}

        scores = [r['consistency_evaluation']['overall_consistency_rate'] for r in valid_samples]
        avg_score = sum(scores) / len(scores) if scores else 0

        total_consistent = sum(r['consistency_evaluation'].get('consistent_claims', 0) for r in valid_samples)
        total_contradictory = sum(r['consistency_evaluation'].get('contradictory_claims', 0) for r in valid_samples)
        total_ambiguous = sum(r['consistency_evaluation'].get('ambiguous_claims', 0) for r in valid_samples)

        return {
            "avg_consistency_rate": avg_score,
            "total_samples": len(results),
            "valid_samples": len(valid_samples),
            "claim_statistics": {
                "total_consistent_claims": total_consistent,
                "total_contradictory_claims": total_contradictory,
                "total_ambiguous_claims": total_ambiguous,
                "avg_consistent_per_sample": total_consistent / len(valid_samples) if valid_samples else 0,
                "avg_contradictory_per_sample": total_contradictory / len(valid_samples) if valid_samples else 0
            },
            "score_distribution": {
                "perfect_0.95_1.0": sum(1 for s in scores if s >= 0.95),
                "high_0.8_0.95": sum(1 for s in scores if 0.8 <= s < 0.95),
                "medium_0.5_0.8": sum(1 for s in scores if 0.5 <= s < 0.8),
                "low_below_0.5": sum(1 for s in scores if s < 0.5)
            }
        }

    @staticmethod
    def print_summary(model_name: str, stats: Dict):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"Consistency Rate Summary - {model_name}")
        print(f"{'='*80}\n")

        if stats['avg_consistency_rate'] is None:
            print("No valid evaluation results")
            return

        print(f"CR Score: {stats['avg_consistency_rate']:.4f}")
        print(f"Valid Samples: {stats['valid_samples']} / {stats['total_samples']}\n")

        print(f"{'─'*80}")
        print(f"Claim Statistics:")
        claim_stats = stats['claim_statistics']
        print(f"  Consistent: {claim_stats['total_consistent_claims']}")
        print(f"  Contradictory: {claim_stats['total_contradictory_claims']}")
        print(f"  Ambiguous: {claim_stats['total_ambiguous_claims']}")

        print(f"\nScore Distribution:")
        for range_name, count in stats['score_distribution'].items():
            print(f"  {range_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Consistency Rate Evaluation using GPT-4o-mini")
    parser.add_argument('--mode', type=str, choices=['comparison', 'responses'], default='responses',
                        help='Evaluation mode')
    parser.add_argument('--response_files', type=str, nargs='+',
                        help='Paths to response JSON files')
    parser.add_argument('--data_dir', type=str, default='./merged_images/multi_diff',
                        help='Base directory containing sample images')
    parser.add_argument('--output_dir', type=str, default='./eval_results_consistency',
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=None,
                        help='OpenAI API base URL')

    args = parser.parse_args()

    evaluator = ConsistencyRateEvaluator(api_key=args.api_key, base_url=args.base_url)

    if args.response_files:
        evaluator.evaluate_multiple_models(
            args.response_files, args.data_dir, args.output_dir, args.max_samples
        )
    else:
        parser.error("--response_files is required")


if __name__ == "__main__":
    main()
