#!/usr/bin/env python3
"""
Visualization utilities for attention maps and activation patterns.
"""

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image


def visualize_token_attentions(raw_image, attentions, token_texts, vision_shape, save_path):
    """Visualize token-wise attention maps."""
    if not attentions:
        return

    # Resize raw image to match vision shape
    img_array = np.array(raw_image)
    h, w = vision_shape
    resized_img = cv2.resize(img_array, (w, h))

    # Create figure
    fig, axes = plt.subplots(1, min(5, len(attentions)), figsize=(20, 4))
    if len(attentions) == 1:
        axes = [axes]

    for i, (att, token) in enumerate(zip(attentions, token_texts[:5])):
        if i >= 5:
            break

        # Reshape attention to vision shape
        reshaped_att = att.reshape(h, w)
        # Upscale to original image size
        upscaled_att = cv2.resize(reshaped_att.numpy(), (img_array.shape[1], img_array.shape[0]))

        # Overlay attention on image
        heatmap = plt.cm.hot(upscaled_att)[:, :, :3]
        overlay = (0.7 * img_array + 0.3 * (heatmap * 255)).astype(np.uint8)

        axes[i].imshow(overlay)
        axes[i].set_title(f'Token: {token[:10]}...', fontsize=10)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_attention_heatmap(attention_weights, save_path, title="Attention Heatmap"):
    """Plot attention weight heatmap."""
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_activation_patterns(activations, save_path, title="Activation Patterns"):
    """Plot neuron activation patterns across layers."""
    layers = list(range(len(activations)))
    avg_activations = [np.mean(layer_act) if layer_act else 0 for layer_act in activations]

    plt.figure(figsize=(12, 6))
    plt.plot(layers, avg_activations, marker='o', linewidth=2, markersize=6)
    plt.title(title)
    plt.xlabel('Layer')
    plt.ylabel('Average Activation')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison_bars(data1, data2, labels, title, save_path):
    """Plot comparison bar charts."""
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, data1, width, label='Group 1', alpha=0.8)
    ax.bar(x + width/2, data2, width, label='Group 2', alpha=0.8)

    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
