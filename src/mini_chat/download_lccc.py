"""
下载 LCCC base 数据集到 src/dataset。

默认下载文件:
    - lccc_base_train.jsonl.gz
    - lccc_base_valid.jsonl.gz
    - lccc_base_test.jsonl.gz
"""
import argparse
import os
import urllib.request


BASE_URL = "https://hf-mirror.com/datasets/silver/lccc/resolve/main"
BASE_FILES = [
    "lccc_base_train.jsonl.gz",
    "lccc_base_valid.jsonl.gz",
    "lccc_base_test.jsonl.gz",
]


def download_file(url: str, output_path: str) -> None:
    """下载单个文件。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\n开始下载: {url}")
    print(f"保存到:   {output_path}")
    urllib.request.urlretrieve(url, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"下载完成: {output_path} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 LCCC base 数据集")
    default_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../dataset"))
    parser.add_argument(
        "--dataset-dir",
        default=default_dataset_dir,
        help="数据集保存目录（默认: src/dataset）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新下载（已存在文件也覆盖）",
    )
    args = parser.parse_args()

    print("目标目录:", args.dataset_dir)
    os.makedirs(args.dataset_dir, exist_ok=True)

    for filename in BASE_FILES:
        url = f"{BASE_URL}/{filename}"
        output_path = os.path.join(args.dataset_dir, filename)

        if os.path.exists(output_path) and not args.force:
            print(f"\n跳过已存在文件: {output_path}")
            continue

        try:
            download_file(url, output_path)
        except Exception as exc:
            print(f"\n下载失败: {filename}")
            print(f"错误信息: {exc}")
            raise

    print("\n全部处理完成。")


if __name__ == "__main__":
    main()

