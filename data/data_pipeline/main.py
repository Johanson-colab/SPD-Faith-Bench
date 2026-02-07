from generator import SpotDifferenceGenerator
from config import (
    COCO_ANN_FILE, COCO_IMG_DIR, 
    EASY_IDS_DIR, MEDIUM_IDS_DIR, HARD_IDS_DIR
)
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def process_single_image(args_tuple):
    """处理单个图像的任务函数"""
    image_id, difficulty, coco_ann_file, coco_img_dir = args_tuple
    
    # 每个线程创建自己的 generator 实例，避免线程安全问题
    generator = SpotDifferenceGenerator(
        coco_ann_file=coco_ann_file,
        coco_img_dir=coco_img_dir
    )
    
    try:
        generator.generate_single_group_1_to_n(image_id, difficulty)
        return {'success': True, 'image_id': image_id, 'difficulty': difficulty}
    except Exception as e:
        return {'success': False, 'image_id': image_id, 'difficulty': difficulty, 'error': str(e)}

def generate_data(workers=1, difficulty_filter=None, start=None, end=None):

    easy_ids = [int(f.stem) for f in EASY_IDS_DIR.glob("*.jpg")]
    medium_ids = [int(f.stem) for f in MEDIUM_IDS_DIR.glob("*.jpg")]
    hard_ids = [int(f.stem) for f in HARD_IDS_DIR.glob("*.jpg")]
    
    # 对每个难度的ID列表进行切片
    if start is not None or end is not None:
        slice_start = start if start is not None else 0
        slice_end = end if end is not None else None
        easy_ids = easy_ids[slice_start:slice_end]
        medium_ids = medium_ids[slice_start:slice_end]
        hard_ids = hard_ids[slice_start:slice_end]
    
    # 根据难度过滤器选择要处理的任务
    tasks = []
    if difficulty_filter is None or difficulty_filter == 'easy':
        tasks.extend([(img_id, 'easy') for img_id in easy_ids])
    if difficulty_filter is None or difficulty_filter == 'medium':
        tasks.extend([(img_id, 'medium') for img_id in medium_ids])
    if difficulty_filter is None or difficulty_filter == 'hard':
        tasks.extend([(img_id, 'hard') for img_id in hard_ids])

    total_tasks = len(tasks)
    
    if total_tasks == 0:
        print("没有找到需要处理的任务！")
        return
    
    # 准备任务参数
    task_args = [
        (image_id, diff, COCO_ANN_FILE, COCO_IMG_DIR)
        for image_id, diff in tasks
    ]
    
    # 线程安全的进度条
    pbar = tqdm(total=total_tasks, desc="总进度", unit="images")
    lock = threading.Lock()
    
    failed_tasks = []
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_image, args): args
            for args in task_args
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                with lock:
                    if result['success']:
                        pbar.update(1)
                    else:
                        failed_tasks.append(result)
                        # 使用tqdm.write()避免打断进度条
                        pbar.write(f"生成图像 {result['image_id']} ({result['difficulty']}) 失败: {result['error']}")
                        pbar.update(1)
            except Exception as e:
                # 如果 future.result() 本身抛出异常（不应该发生，但为了安全起见）
                with lock:
                    failed_tasks.append({
                        'success': False,
                        'image_id': 'unknown',
                        'difficulty': 'unknown',
                        'error': f"任务执行异常: {str(e)}"
                    })
                    # 使用tqdm.write()避免打断进度条
                    pbar.write(f"任务执行异常: {e}")
                    pbar.update(1)
    
    pbar.close()
    
    # 输出总结
    success_count = total_tasks - len(failed_tasks)
    print(f"\n✓ 数据生成完成！")
    print(f"  - 成功: {success_count}/{total_tasks}")
    print(f"  - 失败: {len(failed_tasks)}/{total_tasks}")
    
    # 如果有失败的任务，输出详细信息
    if failed_tasks:
        print(f"\n失败的任务详情:")
        for task in failed_tasks:
            print(f"  - 图像 {task.get('image_id', 'unknown')} ({task.get('difficulty', 'unknown')}): {task.get('error', '未知错误')}")
    

def main():
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        
    )
    
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=1,
        help=''
    )
    
    parser.add_argument(
        '-d', '--difficulty',
        type=str,
        choices=['easy', 'medium', 'hard'],
        default=None,
        help=''
    )
    
    parser.add_argument(
        '-s','--start',
        type=int,
        default=None,
        help=''
    )
    
    parser.add_argument(
        '-e','--end',
        type=int,
        default=None,
        help=''
    )
    
    args = parser.parse_args()
    

    # 执行数据生成
    generate_data(
        workers=args.workers,
        difficulty_filter=args.difficulty,
        start=args.start,
        end=args.end
    )

if __name__ == "__main__":
    main()
