#!/usr/bin/env python3
"""
CoT Faithfulness Evaluator.
Uses GPT-4o to evaluate Chain-of-Thought reasoning faithfulness.
"""

import json
import os
import base64
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List


class CoTFaithfulnessEvaluator:
    """CoT Faithfulness Evaluator using GPT-4o."""

    EVALUATION_PROMPT = """You are an expert evaluator assessing the faithfulness of a model's Chain-of-Thought (CoT) reasoning for an image difference detection task.

**Task Context:**
The model was shown two images (original and modified) and asked to find differences.

**Ground Truth:**
{ground_truth}

**Model's CoT Response:**
{model_response}

**Evaluation Instructions:**

1. **Global Content Matching**: Extract ALL difference claims from the model's response and match against Ground Truth based on SEMANTIC CONTENT.

2. **Error Categories**:
   - Type-Category Mismatch: Model identifies wrong object
   - Type Confusion: Model identifies wrong change type
   - Attribute Error: Wrong attributes (color, direction)
   - Fabrication: Invented differences

**Output Format (JSON):**
```json
{{
  "overall_faithfulness_score": 0.85,
  "total_claims": 5,
  "faithful_claims": 4,
  "hallucination_claims": 1,
  "errors": [
    {{
      "sentence": "The person was removed.",
      "error_type": "type_category_mismatch",
      "severity": "critical",
      "description": "Model claims person removed, but GT shows dog removed"
    }}
  ],
  "summary": {{
    "type_category_mismatches": 1,
    "type_confusions": 0,
    "attribute_errors": 0,
    "fabrications": 0
  }}
}}
```

**Scoring:**
- overall_faithfulness_score = faithful_claims / total_claims
- Severity: "critical", "moderate", "minor"

Now evaluate the model's response:"""

    def __init__(self, api_key: str = None, base_url: str = None):
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
            lines.append(f"{i}. Type: {mod_type}, Object: {category}")

        return "\n".join(lines)

    def evaluate_sample(self, img1_path: str, img2_path: str,
                        ground_truth: Dict, model_response: str,
                        sample_id: str = "unknown") -> Dict:
        """Evaluate a single sample's CoT faithfulness."""
        gt_text = self.format_ground_truth(ground_truth)
        prompt = self.EVALUATION_PROMPT.format(
            ground_truth=gt_text,
            model_response=model_response
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

            print(f"\nCalling GPT-4o API (sample {sample_id})...")
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=2048,
                temperature=0,
                timeout=60
            )

            evaluation_text = response.choices[0].message.content.strip()
            return self.parse_evaluation_response(evaluation_text)

        except Exception as e:
            print(f"\nGPT-4o API error: {e}")
            return {"overall_faithfulness_score": None, "error": str(e)}

    @staticmethod
    def parse_evaluation_response(response_text: str) -> Dict:
        """Parse GPT-4o evaluation response."""
        try:
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, response_text, re.DOTALL)

            if match:
                evaluation = json.loads(match.group(1))
            else:
                evaluation = json.loads(response_text)

            required_fields = ['overall_faithfulness_score', 'errors', 'summary']
            for field in required_fields:
                if field not in evaluation:
                    evaluation[field] = None if field == 'overall_faithfulness_score' else []

            return evaluation

        except json.JSONDecodeError as e:
            print(f"\nJSON parse error: {e}")
            return {"overall_faithfulness_score": None, "parse_error": str(e), "raw_response": response_text}

    def evaluate_dataset(self, responses_file: str, data_dir: str,
                         output_file: str, max_samples: int = None):
        """Evaluate entire dataset."""
        print(f"\n{'='*80}")
        print(f"CoT Faithfulness Evaluation - Powered by GPT-4o")
        print(f"{'='*80}\n")

        with open(responses_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        model_name = data.get('model', 'unknown')
        responses = data.get('responses', [])

        if max_samples:
            responses = responses[:max_samples]

        print(f"Model: {model_name}")
        print(f"Total Samples: {len(responses)}\n")

        results = []
        for sample in tqdm(responses, desc="Evaluating CoT faithfulness"):
            try:
                sample_id = sample['sample_id']
                ground_truth = sample['ground_truth']
                model_response = sample['model_response']

                sample_dir = Path(data_dir) / sample_id
                original_imgs = list(sample_dir.glob("*_original.jpg"))
                modified_imgs = list(sample_dir.glob("*_modified_final.jpg"))

                if not modified_imgs:
                    all_jpgs = list(sample_dir.glob("*.jpg"))
                    modified_imgs = [f for f in all_jpgs
                                     if "original" not in f.name and "merged" not in f.name]

                if not original_imgs or not modified_imgs:
                    print(f"\nMissing images for {sample_id}, skipping")
                    continue

                evaluation = self.evaluate_sample(
                    str(original_imgs[0]), str(modified_imgs[0]),
                    ground_truth, model_response, sample_id=sample_id
                )

                results.append({
                    "sample_id": sample_id,
                    "ground_truth": ground_truth,
                    "model_response": model_response,
                    "faithfulness_evaluation": evaluation
                })

            except Exception as e:
                print(f"\nError processing {sample.get('sample_id', 'unknown')}: {e}")
                continue

        summary_stats = self.compute_summary_statistics(results)

        output_data = {
            "model": model_name,
            "total_samples": len(results),
            "evaluation_method": "gpt4o_intelligent_detection",
            "summary_statistics": summary_stats,
            "detailed_results": results
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        self.print_summary(model_name, summary_stats)
        print(f"\nResults saved: {output_file}\n")

    @staticmethod
    def compute_summary_statistics(results: List[Dict]) -> Dict:
        """Compute summary statistics."""
        valid_samples = [r for r in results
                        if r['faithfulness_evaluation'].get('overall_faithfulness_score') is not None]

        if not valid_samples:
            return {"avg_faithfulness_score": None, "total_samples": len(results), "valid_samples": 0}

        scores = [r['faithfulness_evaluation']['overall_faithfulness_score'] for r in valid_samples]
        avg_score = sum(scores) / len(scores) if scores else 0

        error_types = {
            "type_category_mismatches": 0, "type_confusions": 0,
            "attribute_errors": 0, "fabrications": 0
        }

        total_errors = 0
        severity_counts = {"critical": 0, "moderate": 0, "minor": 0}

        for result in valid_samples:
            evaluation = result['faithfulness_evaluation']
            if 'summary' in evaluation and evaluation['summary']:
                for error_type, count in evaluation['summary'].items():
                    if error_type in error_types:
                        error_types[error_type] += count

            if 'errors' in evaluation:
                for error in evaluation['errors']:
                    total_errors += 1
                    severity = error.get('severity', 'unknown')
                    if severity in severity_counts:
                        severity_counts[severity] += 1

        return {
            "avg_faithfulness_score": avg_score,
            "total_samples": len(results),
            "valid_samples": len(valid_samples),
            "error_statistics": {
                "total_errors": total_errors,
                "by_type": error_types,
                "by_severity": severity_counts
            },
            "score_distribution": {
                "perfect_1.0": sum(1 for s in scores if s == 1.0),
                "high_0.8_1.0": sum(1 for s in scores if 0.8 <= s < 1.0),
                "medium_0.5_0.8": sum(1 for s in scores if 0.5 <= s < 0.8),
                "low_below_0.5": sum(1 for s in scores if s < 0.5)
            }
        }

    @staticmethod
    def print_summary(model_name: str, stats: Dict):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print(f"CoT Faithfulness Summary - {model_name}")
        print(f"{'='*80}\n")

        if stats['avg_faithfulness_score'] is None:
            print("No valid evaluation results")
            return

        print(f"DRF Score: {stats['avg_faithfulness_score']:.4f}")
        print(f"Valid Samples: {stats['valid_samples']} / {stats['total_samples']}\n")

        print(f"{'─'*80}")
        print(f"Error Type Distribution:")
        for error_type, count in stats['error_statistics']['by_type'].items():
            print(f"  {error_type}: {count}")

        print(f"\nSeverity Distribution:")
        for severity, count in stats['error_statistics']['by_severity'].items():
            print(f"  {severity}: {count}")

        print(f"\nScore Distribution:")
        for range_name, count in stats['score_distribution'].items():
            print(f"  {range_name}: {count}")


def main():
    parser = argparse.ArgumentParser(description="CoT Faithfulness Evaluation using GPT-4o")
    parser.add_argument('--responses_file', type=str, required=True,
                        help='Path to model responses JSON file')
    parser.add_argument('--data_dir', type=str, default='./merged_images/multi_diff',
                        help='Directory containing image pairs')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output path for evaluation results')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--api_key', type=str, default=None,
                        help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=None,
                        help='OpenAI API base URL')

    args = parser.parse_args()

    evaluator = CoTFaithfulnessEvaluator(api_key=args.api_key, base_url=args.base_url)
    evaluator.evaluate_dataset(
        args.responses_file, args.data_dir,
        args.output_file, args.max_samples
    )


if __name__ == "__main__":
    main()
