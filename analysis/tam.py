#!/usr/bin/env python3
"""
Token Activation Map (TAM).
Visual explanation method for VLM predictions.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.optimize import minimize_scalar
from PIL import Image, ImageEnhance


def apply_blue_tint(image, intensity=0.6):
    """Apply cool blue/purple tint filter for visual consistency."""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_rgba = image.convert("RGBA")
    blue_tint = Image.new('RGBA', image.size, (30, 0, 80, int(255 * intensity)))
    return Image.blend(image_rgba, blue_tint, alpha=intensity).convert('RGB')


def apply_turbo_colormap(score_map, alpha=0.35):
    """Apply turbo colormap to activation scores."""
    norm = Normalize(vmin=0, vmax=1)
    normalized_score = norm(score_map)

    cmap = plt.cm.turbo
    colored_map = cmap(normalized_score)
    colored_map_bgr = (colored_map[:, :, [2, 1, 0]] * 255).astype(np.uint8)

    return colored_map_bgr, alpha


def rank_gaussian_filter(img, kernel_size=3):
    """Apply rank-based Gaussian-weighted filter for robust denoising.

    Parameters:
        img: Input 2D grayscale image.
        kernel_size: Size of the square kernel (must be odd).

    Returns:
        Denoised image after applying the filter.
    """
    filtered_img = np.zeros_like(img)
    pad_width = kernel_size // 2
    padded_img = np.pad(img, pad_width, mode='reflect')
    ax = np.array(range(kernel_size ** 2)) - kernel_size ** 2 // 2

    for i in range(pad_width, img.shape[0] + pad_width):
        for j in range(pad_width, img.shape[1] + pad_width):
            window = padded_img[i - pad_width:i + pad_width + 1,
                                j - pad_width:j + pad_width + 1]

            sorted_window = np.sort(window.flatten())
            mean = sorted_window.mean()
            if mean > 0:
                sigma = sorted_window.std() / mean
                kernel = np.exp(-(ax ** 2) / (2 * sigma ** 2))
                kernel = kernel / np.sum(kernel)
                value = (sorted_window * kernel).sum()
            else:
                value = 0
            filtered_img[i - pad_width, j - pad_width] = value

    return filtered_img


def least_squares(map1, map2):
    """Find scalar minimizing squared difference between map1 and scalar*map2."""
    def diff(x, map1, map2):
        return np.sum((map1 - map2 * x) ** 2)

    result = minimize_scalar(diff, args=(map1, map2))
    return result.x


def id2idx(inp_id, target_id, return_last=False):
    """Convert target ID to index in input list."""
    if isinstance(target_id, list):
        n = len(target_id)
        indexes = [i for i in range(len(inp_id) - n + 1)
                   if inp_id[i:i + n] == target_id]
        if len(indexes) > 0:
            idx = indexes[-1]
            if return_last:
                idx += len(target_id) - 1
        else:
            idx = -1
    else:
        try:
            idx = inp_id.index(target_id)
        except:
            idx = -1
    return idx


def multimodal_process(raw_img, vision_shape, img_scores, txt_scores, txts,
                       candidates, candi_scores, vis_token_idx, img_save_fn,
                       eval_only=False, vis_width=-1):
    """Process multimodal tokens for visualization.

    Normalizes, filters, and blends image/text activation scores.

    Args:
        raw_img: Raw input image(s).
        vision_shape: Shape of vision tokens.
        img_scores: Activation scores for image tokens.
        txt_scores: Activation scores for text tokens.
        txts: Text tokens to visualize.
        candidates: Top-k prediction candidates.
        candi_scores: Scores for candidates.
        vis_token_idx: Index of token to explain.
        img_save_fn: Path to save visualization.
        eval_only: If True, return only evaluation maps.
        vis_width: Width for resizing (-1 for no resize).

    Returns:
        (out_img, img_map): Visualization image and evaluation map.
    """
    txt_scores = txt_scores[:-1]
    all_scores = np.concatenate([img_scores, txt_scores], 0)
    all_scores = (all_scores - all_scores.min()) / (all_scores.max() - all_scores.min() + 1e-8)
    img_scores = all_scores[:len(img_scores)]
    txt_scores = all_scores[len(img_scores):]

    eval_only = True if img_save_fn == "" else False

    if isinstance(vision_shape[0], tuple):
        # Multiple images
        resized_img, img_map = [], []
        start_idx = 0

        for n in range(len(vision_shape)):
            t_h, t_w = vision_shape[n]
            h, w, c = raw_img[n].shape

            if vis_width > 0:
                h = int(vis_width)
                w = int(float(w) / h * vis_width)

            end_idx = start_idx + int(t_h * t_w)
            img_map_ = rank_gaussian_filter(img_scores[start_idx:end_idx].reshape(t_h, t_w), 3)
            start_idx = end_idx

            img_map_ = (img_map_ * 255).astype('uint8')
            img_map_ = np.clip(img_map_ * 1.3, 0, 255).astype('uint8')

            if not eval_only:
                img_map_norm = img_map_.astype(np.float32) / 255.0
                img_map_, alpha = apply_turbo_colormap(img_map_norm, alpha=0.35)
                img_map_ = cv2.resize(img_map_, (w, h))
                if vis_width > 0:
                    raw_img_ = cv2.resize(raw_img[n], (w, h))
                    resized_img.append(raw_img_)

            img_map.append(img_map_)

        if eval_only:
            return None, img_map

        alpha = 0.35
        out_img = [img_map[i].astype(np.float32) * alpha + resized_img[i].astype(np.float32) * (1 - alpha)
                   for i in range(len(vision_shape))]
        out_img = [img.astype(np.uint8) for img in out_img]
        out_img = np.concatenate(out_img, 1)

        return out_img, img_map

    elif len(vision_shape) == 2:
        # Single image
        t_h, t_w = vision_shape
        h, w, c = raw_img.shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        img_scores = rank_gaussian_filter(img_scores.reshape(t_h, t_w), 3)
        img_scores = (img_scores * 255).astype('uint8')
        img_scores = np.clip(img_scores * 1.3, 0, 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_scores_norm = img_scores.astype(np.float32) / 255.0
        img_map, alpha = apply_turbo_colormap(img_scores_norm, alpha=0.35)
        img_map = cv2.resize(img_map, (w, h))
        if vis_width > 0:
            raw_img = cv2.resize(raw_img, (w, h))

        out_img = img_map.astype(np.float32) * alpha + raw_img.astype(np.float32) * (1 - alpha)
        out_img = out_img.astype(np.uint8)

        return out_img, img_scores

    else:
        # Video batch
        b, t_h, t_w = vision_shape
        h, w, c = raw_img[0].shape
        if vis_width > 0:
            h = int(float(h) / w * vis_width)
            w = int(vis_width)

        img_scores = np.array([rank_gaussian_filter(_.reshape(t_h, t_w), 3)
                               for _ in np.array_split(img_scores, b)])
        img_scores = (img_scores * 255).astype('uint8')
        img_scores = np.clip(img_scores * 1.3, 0, 255).astype('uint8')

        if eval_only:
            return None, img_scores

        img_map = []
        alpha = 0.35
        for score in img_scores:
            score_norm = score.astype(np.float32) / 255.0
            colored_map, _ = apply_turbo_colormap(score_norm, alpha=alpha)
            img_map.append(cv2.resize(colored_map, (w, h)))

        if vis_width > 0:
            raw_img = [cv2.resize(_, (w, h)) for _ in raw_img]

        out_img = [img_map[i].astype(np.float32) * alpha + raw_img[i].astype(np.float32) * (1 - alpha)
                   for i in range(b)]
        out_img = [img.astype(np.uint8) for img in out_img]
        out_img = np.concatenate(out_img, 1)

        return out_img, img_scores


def TAM(tokens, vision_shape, logit_list, special_ids, vision_input,
        processor, save_fn, target_token, img_scores_list, eval_only=False,
        prompt_tracking=None):
    """Generate Token Activation Map (TAM).

    Args:
        tokens: Token sequence including input and generated tokens.
        vision_shape: Shape info of vision input.
        logit_list: List of logits tensors for each generation round.
        special_ids: Dictionary with special token ids (img_id, prompt_id, answer_id).
        vision_input: Raw vision input.
        processor: Model processor.
        save_fn: Path to save visualization.
        target_token: Token index or (round_idx, prompt_token_idx) to explain.
        img_scores_list: List to accumulate image maps for ECI.
        eval_only: Run in evaluation mode.
        prompt_tracking: Optional tracking dict for prompt statistics.

    Returns:
        img_map: The TAM for evaluation.
    """
    import torch

    img_id = special_ids['img_id']
    prompt_id = special_ids['prompt_id']
    answer_id = special_ids['answer_id']

    # Find token indices
    if len(img_id) == 1:
        img_idx = (np.array(tokens) == img_id[0]).nonzero()[0]
    else:
        img_idx = [id2idx(tokens, img_id[0], True), id2idx(tokens, img_id[1])]

    prompt_idx = [id2idx(tokens, prompt_id[0], True), id2idx(tokens, prompt_id[1])]
    answer_idx = [id2idx(tokens, answer_id[0], True), id2idx(tokens, answer_id[1])]

    # Decode tokens
    prompt = processor.tokenizer.tokenize(
        processor.batch_decode([tokens[prompt_idx[0] + 1:prompt_idx[1]]],
                                skip_special_tokens=False)[0]
    )
    answer = processor.tokenizer.tokenize(
        processor.batch_decode([tokens[answer_idx[0] + 1:]],
                                skip_special_tokens=False)[0]
    )
    txt_all = prompt + answer

    # Determine target indices
    round_idx = -1
    this_token_idx = 0

    if isinstance(target_token, int):
        round_idx = target_token
        this_token_idx = -1
        vis_token_idx = len(prompt) + target_token
    else:
        round_idx, prompt_token_idx = target_token
        this_token_idx = prompt_idx[0] + prompt_token_idx + 1
        vis_token_idx = prompt_token_idx

    # Handle round 0 recursion
    if round_idx == 0 and isinstance(target_token, int):
        for t in range(len(prompt) + 1):
            img_map = TAM(tokens, vision_shape, logit_list, special_ids, vision_input,
                          processor, save_fn if t == len(prompt) else '', [0, t],
                          img_scores_list, eval_only, prompt_tracking)
            if t == 0:
                first_ori = img_map
        return first_ori

    # Get target class ID
    if round_idx == 0:
        if isinstance(target_token, tuple):
            _, prompt_token_idx = target_token
            if prompt_token_idx == len(prompt):
                this_token_idx = logit_list[0].shape[1] - 1
                cls_id = tokens[this_token_idx]
            elif prompt_token_idx == 0:
                cls_id = logit_list[0][0, prompt_idx[0] + 1].argmax(0)
            else:
                cls_id = tokens[this_token_idx]
        else:
            this_token_idx = logit_list[0].shape[1] - 1
            cls_id = tokens[this_token_idx]
    else:
        cls_id = tokens[answer_idx[0] + round_idx + 1]

    # Compute scores
    scores = torch.cat([logit_list[_][0, :, cls_id] for _ in range(round_idx + 1)], -1).clip(min=0)
    scores = scores.detach().cpu().float().numpy()

    prompt_scores = scores[prompt_idx[0] + 1:prompt_idx[1]]
    last_prompt = scores[logit_list[0].shape[1] - 1:logit_list[0].shape[1]]
    answer_scores = scores[answer_idx[0] + 1:]
    txt_scores = np.concatenate([prompt_scores, last_prompt, answer_scores], -1)

    # Image scores
    if isinstance(img_idx, list):
        img_scores = scores[img_idx[0] + 1:img_idx[1]]
    else:
        img_scores = scores[img_idx]

    img_scores_list.append(img_scores)

    # Estimated Causal Inference (ECI)
    if len(img_scores_list) > 1 and vis_token_idx < len(txt_all):
        non_repeat_idx = []
        for i in range(min(vis_token_idx, len(img_scores_list))):
            if i < len(txt_all) and txt_all[i] != txt_all[vis_token_idx]:
                non_repeat_idx.append(i)

        if len(non_repeat_idx) > 0:
            txt_scores_ = txt_scores[non_repeat_idx]
            img_scores_list_ = [img_scores_list[_] for _ in non_repeat_idx]

            w = txt_scores_
            w = w / (w.sum() + 1e-8)
            interf_img_scores = (np.stack(img_scores_list_, 0) * w.reshape(-1, 1)).sum(0)

            scaled_map = least_squares(img_scores, interf_img_scores)
            img_scores = (img_scores - interf_img_scores * scaled_map).clip(min=0)

    # Prepare visualization input
    if isinstance(vision_shape[0], tuple):
        cv_img = []
        for img in vision_input:
            pil_img = Image.fromarray(np.array(img))
            tinted_img = apply_blue_tint(pil_img, intensity=0.6)
            tinted_img = ImageEnhance.Brightness(tinted_img).enhance(1.15)
            cv_img.append(cv2.cvtColor(np.array(tinted_img), cv2.COLOR_RGB2BGR))
    elif len(vision_shape) == 2:
        cv_img = np.array(vision_input)
        if len(cv_img.shape) == 4 and cv_img.shape[0] == 1:
            cv_img = cv_img[0]

        pil_img = Image.fromarray(cv_img)
        tinted_img = apply_blue_tint(pil_img, intensity=0.6)
        tinted_img = ImageEnhance.Brightness(tinted_img).enhance(1.15)
        cv_img = cv2.cvtColor(np.array(tinted_img), cv2.COLOR_RGB2BGR)
    else:
        cv_img = []
        for img in vision_input[0]:
            pil_img = Image.fromarray(np.array(img))
            tinted_img = apply_blue_tint(pil_img, intensity=0.6)
            tinted_img = ImageEnhance.Brightness(tinted_img).enhance(1.15)
            cv_img.append(cv2.cvtColor(np.array(tinted_img), cv2.COLOR_RGB2BGR))

    # Get top candidates
    candi_scores, candi_ids = logit_list[round_idx][0, this_token_idx].topk(3)
    candi_scores = candi_scores.softmax(0)
    candidates = processor.batch_decode([[_] for _ in candi_ids])

    # Generate visualization
    vis_img, img_map = multimodal_process(
        cv_img, vision_shape, img_scores, txt_scores, txt_all,
        candidates, candi_scores, vis_token_idx, save_fn,
        eval_only=eval_only, vis_width=-1 if eval_only else 500
    )

    # Save visualization
    if save_fn != '' and vis_token_idx < (len(txt_all) - 1) and isinstance(vis_img, np.ndarray):
        os.makedirs(os.path.dirname(save_fn), exist_ok=True)
        cv2.imwrite(save_fn, vis_img)

    return img_map
