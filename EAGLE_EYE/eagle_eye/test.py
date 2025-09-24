import os
import json
import argparse
from glob import glob

def build_metadata(video_dir, output_file):
    """
    扫描视频目录，生成 metadata.jsonl 文件
    每行一个样本，包含 video_id 和 path
    """
    video_paths = glob(os.path.join(video_dir, "**", "*.mp4"), recursive=True)
    video_paths.sort()  # 保证顺序一致

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, path in enumerate(video_paths):
            item = {
                "video_id": f"{idx:05d}",   # 给每个视频一个统一的ID
                "path": os.path.abspath(path),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Saved metadata with {len(video_paths)} entries to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", type=str, default="/root/autodl-tmp/eagle-eye/EAGLE_EYE/eagle_eye/data",required=True, help="LongVideoBench 视频目录")
    parser.add_argument("--output_file", type=str, default="/root/autodl-tmp/eagle-eye/EAGLE_EYE/eagle_eye/metadata.jsonl", help="输出 JSONL 文件路径")
    args = parser.parse_args()

    build_metadata(args.video_dir, args.output_file)
