import os
import json
import re
import cv2
import numpy as np
from config import COLOR_MAP, PROCESSED_DATA_DIR
from utils import create_mask_from_segmentation, load_coco_image
from api_client import client, gemini_client
from simple_lama_inpainting import SimpleLama
from pycocotools.coco import COCO
from pathlib import Path
from datetime import datetime
from PIL import Image
import base64
from io import BytesIO
# 全局标志，确保GPU检测信息只打印一次
_gpu_info_printed = False

class SpotDifferenceGenerator:
    
    def __init__(self, coco_ann_file, coco_img_dir):
        self.coco = COCO(coco_ann_file)
        self.coco_img_dir = Path(coco_img_dir)
        self.client = client
        self.group_counter = 0
        self.lama_model = None
    
    def _init_lama_model(self):
        global _gpu_info_printed
        if self.lama_model is None:
            # 尝试使用 GPU，如果没有则使用 CPU
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                if cuda_available:
                    device = torch.device('cuda')
                    # 验证GPU是否真的可用
                    try:
                        # 创建一个测试张量来验证GPU
                        test_tensor = torch.zeros(1).to(device)
                        del test_tensor
                        torch.cuda.empty_cache()
                        if not _gpu_info_printed:
                            device_name = torch.cuda.get_device_name(0)
                            print(f"✓ 检测到GPU: {device_name}, 将使用GPU加速")
                            _gpu_info_printed = True
                    except Exception as e:
                        if not _gpu_info_printed:
                            print(f"⚠ GPU检测失败: {e}, 将使用CPU")
                            _gpu_info_printed = True
                        device = torch.device('cpu')
                else:
                    device = torch.device('cpu')
                    if not _gpu_info_printed:
                        print("⚠ 未检测到CUDA设备，将使用CPU")
                        print("  提示：如果您的系统有GPU，请确保：")
                        print("  1. 已安装CUDA版本的PyTorch: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                        print("  2. GPU驱动已正确安装")
                        print("  3. CUDA环境变量已正确设置")
                        _gpu_info_printed = True
                
                # 初始化SimpleLama模型，传入device参数
                self.lama_model = SimpleLama(device=device)
            except Exception as e:
                # 如果导入失败，使用默认设备
                if not _gpu_info_printed:
                    print(f"⚠ 初始化GPU设备失败: {e}, 使用默认设置")
                    _gpu_info_printed = True
                self.lama_model = SimpleLama()
    
    def _pad_to_multiple_of_8(self, img):
        """
        将图像填充到8的倍数，这是 LaMa 模型的最佳实践
        可以提高处理效率和效果
        """
        h, w = img.shape[:2]
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            # 使用边缘反射填充，保持图像内容自然
            img_padded = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            return img_padded, (pad_h, pad_w)
        return img, (0, 0)
    
    def _crop_padding(self, img, pad_h, pad_w):
        """裁剪填充区域"""
        if pad_h > 0 or pad_w > 0:
            h, w = img.shape[:2]
            return img[:h-pad_h if pad_h > 0 else h, :w-pad_w if pad_w > 0 else w]
        return img
    
    def _smooth_mask_edges(self, mask, kernel_size=3):
        """
        平滑 mask 边缘，使修复区域过渡更自然
        这有助于 LaMa 生成更自然的修复结果
        """
        # 对 mask 边缘进行轻微膨胀和模糊，使边缘更柔和
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        # 使用高斯模糊平滑边缘
        mask_smooth = cv2.GaussianBlur(mask_dilated, (kernel_size, kernel_size), 0)
        # 确保 mask 仍然是二值化的（0 和 255）
        _, mask_binary = cv2.threshold(mask_smooth, 127, 255, cv2.THRESH_BINARY)
        return mask_binary
   
    def _ask_llm_color_change(self, img, objects_info, excluded_colors=None):
        """
        询问LLM选择要改变颜色的对象和目标颜色
        
        Args:
            img: 图像
            objects_info: 对象信息列表
            excluded_colors: 已使用过的颜色列表，避免重复使用
        """
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            objects_text = ""
            for i, obj in enumerate(objects_info):
                x, y, w, h = obj['bbox']
                area_percent = obj['area_ratio'] * 100
                objects_text += f"ID{i}: {obj['category']}\n"
                objects_text += f"  - Position: [x={x:.0f}, y={y:.0f}, width={w:.0f}, height={h:.0f}]\n"
                objects_text += f"  - Size: {area_percent:.2f}% of image area\n\n"

            excluded_colors_text = ""
            if excluded_colors:
                excluded_colors_text = f"\nIMPORTANT: The following colors have already been used in previous differences in this image group: {', '.join(excluded_colors)}. Please choose a DIFFERENT color from the available colors list to ensure variety."

            prompt = f"""You are analyzing object annotations for a "spot the difference" puzzle. Below are the objects detected in an image:

            {objects_text}

            Your task:
            1. Select ONE object that would be suitable for color change. Consider:
            - The object should have appropriate size (not too small, not too large)
            - The object should be something that commonly appears in multiple colors in real life
                Examples of good choices: cars, umbrellas, clothing, bags, bicycles, trucks, buses
                Examples of bad choices: grass, sky, bananas, trees (typically one color)
            - The color change should be realistic and follow common sense

            2. Determine what color to change it to:
            - Available colors: red, orange, yellow, lime, green, cyan, blue, purple, pink, magenta
            - Choose a target color that creates a noticeable but realistic change
            - The target color should be different from the object's current color
            {excluded_colors_text}

            IMPORTANT: Return ONLY a valid JSON object (no markdown code blocks, no extra text) in this exact format:
            {{
                "selected_object_id": <object ID number>,
                "object_name": "<category name>",
                "original_color": "<original color name>",
                "target_color": "<color name from the list above>"
            }}

            Make sure the selected object is something that can realistically have different colors in everyday life."""

            message = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            
            response_text = message.choices[0].message.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            return result
        except Exception as e:
            # 静默处理错误，避免打断进度条
            import traceback
            traceback.print_exc()
            return None

    def _ask_llm_remove_object(self, img, objects_info):
        try:
            # 在图像上绘制所有对象的bbox与ID，便于模型参考
            img_annotated = img.copy()
            for i, obj_info in enumerate(objects_info):
                x, y, w, h = obj_info['bbox']
                cv2.rectangle(img_annotated, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                label = f"ID{i}: {obj_info['category']}"
                cv2.putText(img_annotated, label, (int(x), int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            # 汇总对象信息，包含位置与面积比例
            objects_text = "\n".join([
                f"ID{i}: {obj['category']} at position [x={obj['bbox'][0]:.0f}, y={obj['bbox'][1]:.0f}, w={obj['bbox'][2]:.0f}, h={obj['bbox'][3]:.0f}] | size={obj['area_ratio']*100:.2f}%"
                for i, obj in enumerate(objects_info)
            ])

            # 计算面积中位数，向模型提供“适中尺寸”的参考
            area_ratios = [obj['area_ratio'] for obj in objects_info]
            median_area_pct = float(np.median(area_ratios) * 100.0) if area_ratios else 0.0

            prompt = f"""You are analyzing an image for a "spot the difference" puzzle.
The image contains the following objects (marked with green boxes and IDs):

{objects_text}

Your task:
1) Select ONE object to REMOVE from the image.
2) Prefer a moderately sized, non-salient foreground object (roughly near the median area: ~{median_area_pct:.2f}% of the image area).
3) Avoid removing background/structural surfaces (e.g., sky, grass, ground, walls, road, large tables) or extremely large regions that would break scene plausibility.
4) Avoid tiny or heavily occluded objects that are too inconspicuous.
5) The choice should be physically plausible: removing it should not violate basic scene integrity.

IMPORTANT: Return ONLY a valid JSON object (no markdown code blocks, no extra text) in this exact format:
{{
    "selected_object_id": <object ID number>,
    "object_name": "<category name>",
    "reason": "<brief reason for removing this object>"
}}"""

            message = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )

            response_text = message.choices[0].message.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
            return result
        except Exception as e:
            # 静默处理错误，避免打断进度条
            import traceback
            traceback.print_exc()
            return None

    def _ask_vlm_position_change(self, img, objects_info):
        try:
            img_annotated = img.copy()
            for i, obj_info in enumerate(objects_info):
                x, y, w, h = obj_info['bbox']
                cv2.rectangle(img_annotated, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                label = f"ID{i}: {obj_info['category']}"
                cv2.putText(img_annotated, label, (int(x), int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            img_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            buffered = BytesIO()
            pil_img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()

            objects_text = "\n".join([
                f"ID{i}: {obj['category']} at position [x={obj['bbox'][0]:.0f}, y={obj['bbox'][1]:.0f}, w={obj['bbox'][2]:.0f}, h={obj['bbox'][3]:.0f}]"
                for i, obj in enumerate(objects_info)
            ])

            prompt = f"""You are analyzing an image for a "spot the difference" puzzle. The image contains the following objects (marked with green boxes and IDs):

            {objects_text}

            Your task:
            1. Select ONE object suitable for a position change (prefer smaller, movable objects).
            2. Determine its current position.
            3. Suggest a reasonable new position that is visually plausible and creates a noticeable but natural difference.

            Hard constraints:
            - new_bbox MUST have the SAME width and height as original_bbox (only x and y change).
            - new_bbox MUST be fully inside the image boundaries.
            - new_bbox MUST NOT overlap or intersect with ANY other object's bounding box.
            - The move MUST be physically plausible: keep the object on a realistic support surface (e.g., floor/ground/road/table), avoid floating in mid-air, avoid penetrating other objects, and respect perspective/scale.

            Additional guidance:
            - Avoid areas densely occupied by other objects or heavy occlusion.
            - Prefer positions that are clear of other bounding boxes and look natural for the chosen object.

            IMPORTANT: Return ONLY a valid JSON object (no markdown code blocks, no extra text) in this exact format:
            {{
                "selected_object_id": <object ID number>,
                "object_name": "<category name>",
                "original_bbox": [x, y, w, h],
                "new_bbox": [new_x, new_y, w, h],
                "reason": "<brief reason for this position change>"
            }}"""

            message = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
            
            response_text = message.choices[0].message.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            return result
        except Exception as e:
            # 静默处理错误，避免打断进度条
            import traceback
            traceback.print_exc()
            return None
    

    
    def _get_image_and_annotations(self, image_id, difficulty=None):
        """
        根据image_id（去除前缀零的数字）获取图像和所有标注
        
        Args:
            image_id: 图像ID
            difficulty: 难度级别 ('easy', 'medium', 'hard')，用于确定图片所在目录
        """
        file_name = f"{int(image_id):012d}.jpg"
        img_ids = self.coco.getImgIds()
        coco_img_id = None
        for img_id in img_ids:
            img_info = self.coco.loadImgs(img_id)[0]
            if img_info['file_name'] == file_name:
                coco_img_id = img_id
                break
        
        if coco_img_id is None:
            raise ValueError(f"未找到image_id为 {image_id} 的图像")
        
        img_info = self.coco.loadImgs(coco_img_id)[0]
        img = load_coco_image(self.coco_img_dir, img_info)
        if img is None:
            raise ValueError(f"无法加载图像: {file_name}")
        
        ann_ids = self.coco.getAnnIds(imgIds=coco_img_id)
        anns = self.coco.loadAnns(ann_ids)
        

        return img, img_info, anns
    def test_get_image_and_annotations(self, image_id):
        img, img_info, anns = self._get_image_and_annotations(image_id)
        print("img:",img)
        print("img_info:",img_info)
        print("anns:",anns)
        return "ok"

    # Method 1:remove object
    def _remove_object(self, image_id, excluded_indices=None):
        """
        移除对象
        
        Args:
            image_id: 图像ID
            excluded_indices: 已处理过的对象索引列表，避免重复处理
        """
        img, img_info, anns = self._get_image_and_annotations(image_id)
        
        if not anns:
            raise ValueError(f"图像 {image_id} 没有标注信息")
        
        # 过滤掉已处理的对象
        if excluded_indices is None:
            excluded_indices = []
        available_anns = [ann for i, ann in enumerate(anns) if i not in excluded_indices]
        
        if not available_anns:
            raise ValueError(f"图像 {image_id} 没有可用的未处理对象（所有对象都已被处理）")
        
        # 构建可用对象的索引映射
        available_indices = [i for i in range(len(anns)) if i not in excluded_indices]
        
        # 先构建 objects_info 供 LLM 选择（只包含可用对象）
        img_area = img.shape[0] * img.shape[1]
        objects_info = []
        for ann in available_anns:
            cat_info = self.coco.loadCats(ann['category_id'])[0]
            objects_info.append({
                'category': cat_info['name'],
                'bbox': ann['bbox'],
                'area_ratio': ann['area'] / img_area
            })

        # 调用LLM选择移除对象，失败则回退到"面积中位数"策略
        llm_choice = self._ask_llm_remove_object(img, objects_info)
        selected_ann = None
        selection_reason = None
        if llm_choice is not None:
            try:
                idx = int(llm_choice.get('selected_object_id'))
                if 0 <= idx < len(available_anns):
                    # 映射回原始索引
                    original_idx = available_indices[idx]
                    selected_ann = anns[original_idx]
                    selected_original_idx = original_idx
                    selection_reason = llm_choice.get('reason')
                else:
                    # 静默处理，使用回退策略
                    selected_ann = None
            except Exception as e:
                # 静默处理，使用回退策略
                selected_ann = None

        if selected_ann is None:
            # 回退策略：从可用对象中选择面积中位数
            areas = [ann['area'] for ann in available_anns]
            median_area = np.median(areas)
            selected_idx_in_available = min(range(len(available_anns)), 
                                           key=lambda i: abs(available_anns[i]['area'] - median_area))
            selected_original_idx = available_indices[selected_idx_in_available]
            selected_ann = anns[selected_original_idx]
        
        # 获取类别信息
        cat_info = self.coco.loadCats(selected_ann['category_id'])[0]
        
        # 创建mask
        mask = create_mask_from_segmentation(img.shape, selected_ann['segmentation'])
        
        # 平滑 mask 边缘，使修复结果更自然
        mask = self._smooth_mask_edges(mask, kernel_size=3)
        
        # 执行去除操作
        self._init_lama_model()
        
        # 将图像和 mask 填充到8的倍数（LaMa 最佳实践）
        img_padded, (pad_h, pad_w) = self._pad_to_multiple_of_8(img)
        mask_padded, _ = self._pad_to_multiple_of_8(mask)
        
        # 转换为 RGB 和 PIL Image
        img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        mask_pil = Image.fromarray(mask_padded)
        
        # 使用 LaMa 进行修复
        result_pil = self.lama_model(img_pil, mask_pil)
        
        # 转换回 numpy 数组和 BGR
        result_rgb = np.array(result_pil)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        # 裁剪填充区域，恢复原始尺寸
        result_bgr = self._crop_padding(result_bgr, pad_h, pad_w)
        
        # 返回图片和日志信息
        log = {
            'type': 'remove',
            'bbox': [float(x) for x in selected_ann['bbox']],
            'category': cat_info['name'],
            'category_id': int(selected_ann['category_id']),
            'area': float(selected_ann['area']),
            'selection_reason': selection_reason if selection_reason else 'fallback: median-area selection',
            'object_index': selected_original_idx  # 记录处理的对象索引
        }
        
        return {'image': result_bgr, 'log': log, 'processed_index': selected_original_idx}
    
    # Method 2:change object's color
    def _change_object_color(self, image_id, excluded_indices=None, used_colors=None):
        """
        改变对象颜色
        
        Args:
            image_id: 图像ID
            excluded_indices: 已处理过的对象索引列表，避免重复处理
            used_colors: 已使用过的颜色列表，避免重复使用相同颜色
        """
        img, img_info, anns = self._get_image_and_annotations(image_id)
        
        if not anns:
            raise ValueError(f"图像 {image_id} 没有标注信息")
        
        # 过滤掉已处理的对象
        if excluded_indices is None:
            excluded_indices = []
        available_anns = [ann for i, ann in enumerate(anns) if i not in excluded_indices]
        
        if not available_anns:
            raise ValueError(f"图像 {image_id} 没有可用的未处理对象（所有对象都已被处理）")
        
        # 构建可用对象的索引映射
        available_indices = [i for i in range(len(anns)) if i not in excluded_indices]
        
        # 构建objects_info用于LLM（只包含可用对象）
        img_area = img.shape[0] * img.shape[1]
        objects_info = []
        for ann in available_anns:
            cat_info = self.coco.loadCats(ann['category_id'])[0]
            objects_info.append({
                'category': cat_info['name'],
                'bbox': ann['bbox'],
                'area_ratio': ann['area'] / img_area
            })
        
        # 如果提供了已使用的颜色，在提示词中告知LLM避免使用
        if used_colors is None:
            used_colors = []

        # 调用LLM选择物体和目标颜色
        llm_result = self._ask_llm_color_change(img, objects_info, excluded_colors=used_colors)

        if llm_result is None:
            raise ValueError("LLM调用失败，无法选择物体和颜色")
        
        try:
            selected_idx = int(llm_result['selected_object_id'])
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"LLM返回的物体ID格式错误: {e}")
        
        original_color = llm_result['original_color']
        target_color = llm_result['target_color']
        
        if selected_idx < 0 or selected_idx >= len(available_anns):
            raise ValueError(f"选择的物体ID {selected_idx} 超出范围 (0-{len(available_anns)-1})")
        
        # 映射回原始索引
        original_idx = available_indices[selected_idx]
        selected_ann = anns[original_idx]
        cat_info = self.coco.loadCats(selected_ann['category_id'])[0]
        mask = create_mask_from_segmentation(img.shape, selected_ann['segmentation'])
        
        # 执行颜色改变操作
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        mask_bool = mask > 127
        

        if target_color.lower() not in COLOR_MAP:
            raise ValueError(f"不支持的颜色: {target_color}. 支持的颜色: {list(COLOR_MAP.keys())}")
        
        target_hue = COLOR_MAP[target_color.lower()]
        img_hsv[mask_bool, 0] = target_hue
        
        img_hsv = np.clip(img_hsv, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        
        # 返回图片和日志信息
        log = {
            'type': 'color',
            'bbox': [float(x) for x in selected_ann['bbox']],
            'category': cat_info['name'],
            'category_id': int(selected_ann['category_id']),
            'original_color': original_color,
            'target_color': target_color,
            'area': float(selected_ann['area']),
            'object_index': original_idx  # 记录处理的对象索引
        }
        
        return {'image': result, 'log': log, 'processed_index': original_idx, 'used_color': target_color.lower()}

    # Method 3:change object's position
    def _change_object_position(self, image_id, excluded_indices=None):
        """
        改变对象位置
        
        Args:
            image_id: 图像ID
            excluded_indices: 已处理过的对象索引列表，避免重复处理
        """
        img, img_info, anns = self._get_image_and_annotations(image_id)
        
        if not anns:
            raise ValueError(f"图像 {image_id} 没有标注信息")
        
        # 过滤掉已处理的对象
        if excluded_indices is None:
            excluded_indices = []
        available_anns = [ann for i, ann in enumerate(anns) if i not in excluded_indices]
        
        if not available_anns:
            raise ValueError(f"图像 {image_id} 没有可用的未处理对象（所有对象都已被处理）")
        
        # 构建可用对象的索引映射
        available_indices = [i for i in range(len(anns)) if i not in excluded_indices]
        
        # 构建objects_info用于VLM（只包含可用对象）
        objects_info = []
        for ann in available_anns:
            cat_info = self.coco.loadCats(ann['category_id'])[0]
            objects_info.append({
                'category': cat_info['name'],
                'bbox': ann['bbox']
            })
        
        # 调用VLM选择物体和目标位置
        vlm_result = self._ask_vlm_position_change(img, objects_info)
        if vlm_result is None:
            raise ValueError("VLM调用失败，无法选择物体和位置")
        
        try:
            selected_idx = int(vlm_result['selected_object_id'])
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"VLM返回的物体ID格式错误: {e}")
        
        new_bbox = vlm_result['new_bbox']
        
        if selected_idx < 0 or selected_idx >= len(available_anns):
            raise ValueError(f"选择的物体ID {selected_idx} 超出范围 (0-{len(available_anns)-1})")
        
        # 映射回原始索引
        original_idx = available_indices[selected_idx]
        selected_ann = anns[original_idx]
        cat_info = self.coco.loadCats(selected_ann['category_id'])[0]
        mask = create_mask_from_segmentation(img.shape, selected_ann['segmentation'])
        original_bbox = selected_ann['bbox']  # 使用标注中的bbox
        
        try:
            x1, y1, w, h = [int(v) for v in original_bbox]
            x2, y2 = int(new_bbox[0]), int(new_bbox[1])
            
            img_h, img_w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(0, x2), max(0, y2)
            w = min(w, img_w - x1, img_w - x2)
            h = min(h, img_h - y1, img_h - y2)
            
            # 检查裁剪后的尺寸是否有效
            if w <= 0 or h <= 0:
                raise ValueError(f"裁剪后的物体尺寸无效: w={w}, h={h}, 原始bbox={original_bbox}, 新bbox={new_bbox}")
            
            object_region = img[y1:y1+h, x1:x1+w].copy()
            object_mask = mask[y1:y1+h, x1:x1+w].copy()
            
            # 检查 object_mask 是否为空
            if object_mask.size == 0 or object_mask.shape[0] == 0 or object_mask.shape[1] == 0:
                raise ValueError(f"object_mask 为空: shape={object_mask.shape}, y1={y1}, x1={x1}, h={h}, w={w}")
            
            # 检查 object_mask 是否有非零像素
            if np.sum(object_mask > 0) == 0:
                raise ValueError(f"object_mask 中没有非零像素，无法进行物体移动")
            
            self._init_lama_model()
            
            # 平滑 mask 边缘
            mask_smooth = self._smooth_mask_edges(mask, kernel_size=3)
            
            # 将图像和 mask 填充到8的倍数
            img_padded, (pad_h, pad_w) = self._pad_to_multiple_of_8(img.copy())
            mask_padded, _ = self._pad_to_multiple_of_8(mask_smooth)
            
            # 转换为 RGB 和 PIL Image
            img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            mask_pil = Image.fromarray(mask_padded)
            
            # 使用 LaMa 进行修复
            img_removed_pil = self.lama_model(img_pil, mask_pil)
            img_removed_rgb = np.array(img_removed_pil)
            img_removed = cv2.cvtColor(img_removed_rgb, cv2.COLOR_RGB2BGR)
            
            # 裁剪填充区域
            img_removed = self._crop_padding(img_removed, pad_h, pad_w)
            
            result = img_removed.copy()
            
            object_mask_3c = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)
            alpha = object_mask_3c.astype(float) / 255.0
            
            alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
            
            for c in range(3):
                result[y2:y2+h, x2:x2+w, c] = (
                    alpha[:, :, c] * object_region[:, :, c] + 
                    (1 - alpha[:, :, c]) * result[y2:y2+h, x2:x2+w, c]
                )
            
            edge_mask = np.zeros_like(mask)
            edge_mask[y2:y2+h, x2:x2+w] = object_mask
            
            kernel = np.ones((5, 5), np.uint8)
            edge_dilated = cv2.dilate(edge_mask, kernel, iterations=2)
            edge_only = cv2.subtract(edge_dilated, edge_mask)
            
            if edge_only.sum() > 0:
                # 平滑边缘 mask
                edge_only_smooth = self._smooth_mask_edges(edge_only, kernel_size=3)
                
                # 填充到8的倍数
                result_padded, (pad_h, pad_w) = self._pad_to_multiple_of_8(result)
                edge_mask_padded, _ = self._pad_to_multiple_of_8(edge_only_smooth)
                
                # 转换为 RGB 和 PIL Image
                edge_rgb = cv2.cvtColor(result_padded, cv2.COLOR_BGR2RGB)
                edge_pil = Image.fromarray(edge_rgb)
                edge_mask_pil = Image.fromarray(edge_mask_padded)
                
                # 使用 LaMa 修复边缘
                result_pil = self.lama_model(edge_pil, edge_mask_pil)
                result_rgb = np.array(result_pil)
                result = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                
                # 裁剪填充区域
                result = self._crop_padding(result, pad_h, pad_w)
            
            # 返回图片和日志信息
            log = {
                'type': 'position',
                'original_bbox': [float(x) for x in original_bbox],
                'new_bbox': [float(x) for x in new_bbox],
                'category': cat_info['name'],
                'category_id': int(selected_ann['category_id']),
                'area': float(selected_ann['area']),
                'object_index': original_idx  # 记录处理的对象索引
            }
            
            return {'image': result, 'log': log, 'processed_index': original_idx}
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {'image': img, 'log': {'type': 'position', 'error': str(e)}}

    def _ask_llm_difference_type(self, img, anns, num_differences):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # 计算图像总面积和每个对象的面积比例
        img_area = img.shape[0] * img.shape[1]
        objects_info = []
        for ann in anns:
            cat_info = self.coco.loadCats(ann['category_id'])[0]
            area_ratio = ann['area'] / img_area
            objects_info.append({
                'category': cat_info['name'],
                'bbox': ann['bbox'],
                'area_ratio': area_ratio
            })
        
        # 构建对象信息文本，包含面积比例
        objects_text = "\n".join([
            f"ID{i}: {obj['category']} at position [x={obj['bbox'][0]:.0f}, y={obj['bbox'][1]:.0f}, w={obj['bbox'][2]:.0f}, h={obj['bbox'][3]:.0f}] | size={obj['area_ratio']*100:.2f}% of image"
            for i, obj in enumerate(objects_info)
        ])

        prompt = f"""You are analyzing an image for a "spot the difference" puzzle. Your task is to select {num_differences} difference types that will be applied to objects in this image.

        The image contains the following objects (with their bounding boxes and sizes):
        {objects_text}

        Available difference types:
        1. "remove" - Remove an object from the image (using inpainting to fill the gap)
        2. "color" - Change an object's color to a different color
        3. "position" - Move an object to a different position in the image

        CRITICAL SELECTION RULES:

        1. NATURALNESS AND REALISM (Highest Priority):
        - The modified image MUST look natural and realistic, as if it could exist in real life
        - All changes must be physically plausible and consistent with everyday common sense
        - Avoid changes that would break scene coherence or violate basic physics
        - Consider the context: changes should make sense for the scene type (indoor/outdoor, urban/nature, etc.)

        2. SIZE-BASED SELECTION STRATEGY:
        - Large objects:
            * AVOID using "remove" for large objects - removing them would create unnatural large gaps
            * PREFER "color" for large objects - color changes are natural and maintain scene integrity
            * Only use "position" for large objects if the move is physically plausible (e.g., moving furniture)
        
        - Medium objects:
            * Can use "remove", "color", or "position" depending on context
            * Prefer "color" or "position" over "remove" when possible
        
        - Small objects:
            * All three types are acceptable
            * "remove" is most natural for small, non-essential objects

        3. OBJECT TYPE CONSIDERATIONS:
        - Background elements (sky, ground, walls): NEVER use "remove", prefer "color" if applicable
        - Structural elements (buildings, large furniture): Avoid "remove", prefer "color"
        - Movable objects (vehicles, people, small items): "position" is often most natural
        - Objects that commonly appear in multiple colors (cars, clothing, bags): "color" is ideal
        - Decorative or accessory items: "remove" can be natural

        4. DIVERSITY REQUIREMENT:
        - Select {num_differences} DIFFERENT difference types (you can repeat types if needed, but diversity is preferred)
        - Try to use a mix of types when possible to create varied differences

        5. VISIBILITY AND NOTICEABILITY:
        - All differences must be clearly visible to the human eye
        - Differences should be noticeable but not jarring
        - Consider contrast: changes should stand out from the background

        6. SCENE CONSISTENCY:
        - All changes together should create a coherent modified scene
        - Avoid conflicting changes (e.g., removing an object and changing its color)
        - Consider how changes interact with each other

        SELECTION PROCESS:
        1. First, identify which objects are large (check the size percentages)
        2. For large objects, prioritize "color" over "remove"
        3. Consider the object type and its role in the scene
        4. Ensure all selected difference types will result in natural, realistic modifications
        5. Verify that the combination of changes makes sense together

        IMPORTANT: Return ONLY a valid JSON object (no markdown code blocks, no extra text) in this exact format:
        {{
            "differences": ["diff_type1", "diff_type2", "diff_type3", ...]
        }}

        CRITICAL: 
        - The "differences" array MUST contain exactly {num_differences} items
        - Each diff_type MUST be one of: "remove", "color", "position" 
        - Prioritize naturalness and realism above all else
        - Remember: large objects should generally use "color" instead of "remove"
        """
        message = self.client.chat.completions.create(
                model="gemini-2.5-flash",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            )
        response_text = message.choices[0].message.content.strip()
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        result = json.loads(response_text)
        return result
    def generate_single_group(self, image_id, complexity='easy'):
        """
        生成单个图像组，包含多个差异
        
        Args:
            image_id: 图像ID ('5114' or '00005114')
            complexity: 复杂度 ('easy', 'medium', 'hard')
        
        Returns:
            保存的文件夹路径
        """
        if complexity.lower() not in ['easy', 'medium', 'hard']:
            raise ValueError(f"不支持的复杂度: {complexity}")
        
        # 根据复杂度确定差异数量
        complexity_config = {
            'easy': 1,
            'medium': 3,
            'hard': 5
        }
        num_differences = complexity_config[complexity.lower()]
        
        # 获取图像和标注信息，传递complexity参数以确定图片所在目录
        img, img_info, anns = self._get_image_and_annotations(image_id, difficulty=complexity)
        
        if not anns:
            raise ValueError(f"图像 {image_id} 没有标注信息")
        
        # LLM选择差异类型
        llm_result = self._ask_llm_difference_type(img, anns, num_differences)
        diff_types = llm_result['differences']
        
        # 创建输出目录: data/processed/{complexity}/{image_id}/

        output_dir = PROCESSED_DATA_DIR / complexity.lower() / str(image_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存原图
        output_image_path = output_dir / f"{image_id}_original.jpg"
        cv2.imwrite(str(output_image_path), img)
        
        # 存储所有差异的日志
        all_logs = []
        
        # 跟踪已处理的对象索引和已使用的颜色，确保同一组中相同类型的操作处理不同的对象
        processed_indices = []  # 已处理过的对象索引
        used_colors = []  # 已使用过的颜色（用于color操作）
        
        # 依次生成每个差异
        for idx, diff_type in enumerate(diff_types, start=1):
            try:
                if diff_type == 'remove':
                    result = self._remove_object(image_id, excluded_indices=processed_indices)
                    # 记录已处理的对象索引
                    if 'processed_index' in result:
                        processed_indices.append(result['processed_index'])
                        
                elif diff_type == 'color':
                    result = self._change_object_color(image_id, excluded_indices=processed_indices, used_colors=used_colors)
                    # 记录已处理的对象索引和使用的颜色
                    if 'processed_index' in result:
                        processed_indices.append(result['processed_index'])
                    if 'used_color' in result:
                        used_colors.append(result['used_color'])
                        
                elif diff_type == 'position':
                    result = self._change_object_position(image_id, excluded_indices=processed_indices)
                    # 记录已处理的对象索引
                    if 'processed_index' in result:
                        processed_indices.append(result['processed_index'])
                else:
                    raise ValueError(f"不支持的差异类型: {diff_type}")
                
                # 保存修改后的图片
                output_image_path = output_dir / f"{image_id}_modified_diff_{idx}.jpg"
                cv2.imwrite(str(output_image_path), result['image'])
                
                # 添加差异索引到日志
                log_entry = result['log'].copy()
                log_entry['diff_index'] = idx
                all_logs.append(log_entry)
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                # 记录错误到日志
                all_logs.append({
                    'diff_index': idx,
                    'type': diff_type,
                    'error': str(e)
                })
        
        # 保存日志文件
        log_data = {
            'image_id': image_id,
            'complexity': complexity.lower(),
            'num_differences': len(diff_types),
            'differences': all_logs,
            'timestamp': datetime.now().isoformat()
        }
        
        log_path = output_dir / 'log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return output_dir
    def generate_single_group_1_to_n(self, image_id, complexity='medium'):
        """
        生成单个图像组，每组两张图，第一张图是原图，第二张图是修改后的图（包含多个差异），用文生图的方式做
        
        Args:
            image_id: 图像ID ('5114' or '00005114')
            complexity: 复杂度 ('medium', 'hard')
        
        Returns:
            保存的文件夹路径
        """
        if complexity.lower() not in ['medium', 'hard']:
            raise ValueError(f"不支持的复杂度: {complexity}")
        
        # 根据复杂度确定差异数量
        num_differences = 3 if complexity.lower() == 'medium' else 5
        
        # 获取图像（只取图像，不需要标注信息）
        img, _, _ = self._get_image_and_annotations(image_id, difficulty=complexity)
        
        # 获取原图分辨率
        original_height, original_width = img.shape[:2]
        
        # 转换为PIL Image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 创建输出目录: data/processed/{complexity}-gemini/{image_id}/
        output_dir = PROCESSED_DATA_DIR / f"{complexity.lower()}-gemini" / str(image_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原图
        cv2.imwrite(str(output_dir / f"{image_id}_original.jpg"), img)
        
        # 构建prompt，要求模型生成包含多个差异的图片并返回bbox信息
        prompt = f"""You are creating a "spot the difference" puzzle dataset. Your task is to create a modified version of this image with exactly {num_differences} differences.

CRITICAL REQUIREMENTS:
1. The modified image should be identical to the original EXCEPT for {num_differences} small, localized changes
2. Each difference must be confined to a small region - the rest of the image must remain exactly the same
3. You can ONLY make changes using these three methods:
   - REMOVE: Remove a small object from the image. Do NOT remove large objects, background elements, or structural components (like walls, sky, ground, large furniture). Only remove small, foreground objects that can be plausibly absent.
   - COLOR: Change the color of ONE object to a different but realistic color. The new color must be something that object could naturally have in real life (e.g., a red car could be blue, but a person's skin should not be green).
   - POSITION: Move ONE object to a different position. The object must remain on a realistic support surface (floor, table, etc.) and cannot overlap with other objects or float in mid-air.

4. After generating the modified image, you must provide a JSON object listing all {num_differences} differences with their exact bounding boxes.

IMPORTANT - Bounding Box Format:
- ALL bounding boxes MUST use the format: [x, y, width, height]
- (x, y) represents the TOP-LEFT corner coordinates of the bounding box
- width and height are the dimensions of the bounding box
- All coordinates are in pixels, starting from (0, 0) at the top-left corner of the image
- DO NOT use [x1, y1, x2, y2] format (bottom-right corner format)
- DO NOT use center point format [cx, cy, width, height]

Return the response in this format:
1. First, show the modified image
2. Then, provide a JSON object (no markdown code blocks, if the type is position, you must provide the original_bbox and new_bbox while if the type is remove or color, you should provide only one bbox) with this exact structure:
{{
    "differences": [
        {{
            "type": "remove|color",
            "bbox": [x, y, width, height],
            "description": "brief description of what was changed"
        }},
        {{
            "type": "position",
            "original_bbox": [x, y, width, height],
            "new_bbox": [x, y, width, height],
            "description": "brief description of what was changed"
        }},
        ...
    ]
}}

Remember: The differences should be subtle enough to require careful observation, but clear enough to be identifiable. Most of the image must remain unchanged."""

        # 调用gemini-2.5-flash-image模型
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, pil_img],
        )
        
        # 提取生成的图片和文本（包含bbox信息）
        generated_image = None
        bbox_text = ""
        
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if part.text:
                    bbox_text += part.text + "\n"
                elif part.inline_data:
                    image_data = part.inline_data.data
                    generated_image = Image.open(BytesIO(image_data))
        
        if generated_image is None:
            raise ValueError("模型未返回生成的图片")
        
        # 检查并调整生成图片的分辨率，确保与原图一致
        generated_width, generated_height = generated_image.size
        if generated_width != original_width or generated_height != original_height:
            # 使用高质量的重采样方法调整分辨率
            generated_image = generated_image.resize(
                (original_width, original_height), 
                Image.Resampling.LANCZOS
            )
        
        # 保存修改后的图片
        generated_image.save(str(output_dir / f"{image_id}_modified.jpg"))
        
        # 解析bbox信息
        differences = []
        if bbox_text:
            try:
                # 尝试提取JSON部分
                if "```json" in bbox_text:
                    bbox_text = bbox_text.split("```json")[1].split("```")[0].strip()
                elif "```" in bbox_text:
                    bbox_text = bbox_text.split("```")[1].split("```")[0].strip()
                
                # 尝试找到JSON对象
                json_match = re.search(r'\{.*\}', bbox_text, re.DOTALL)
                if json_match:
                    bbox_data = json.loads(json_match.group())
                    differences = bbox_data.get('differences', [])
            except Exception as e:
                print(f"解析bbox信息失败: {e}")
                differences = [{"type": "unknown", "bbox": None, "description": bbox_text}]
        
        # 保存日志文件
        log_data = {
            'image_id': image_id,
            'complexity': complexity.lower(),
            'num_differences': num_differences,
            'differences': differences,
            'timestamp': datetime.now().isoformat()
        }
        
        log_path = output_dir / 'log.json'
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
        
        return output_dir



  