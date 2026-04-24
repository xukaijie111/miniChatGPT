"""
LCCC 数据集下载脚本
使用国内镜像下载
"""
import os
import gzip
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(__file__), "dataset")
MIRROR = "https://hf-mirror.com/datasets/silver/lccc/resolve/main"

FILES = {
    "base": [
        "lccc_base_train.jsonl.gz",
        "lccc_base_valid.jsonl.gz",
        "lccc_base_test.jsonl.gz",
    ],
    "large": [
        "lccc_large.jsonl.gz",
    ],
}


def download_file(url: str, output_path: str):
    """下载文件"""
    print(f"正在下载: {url}")
    print(f"保存到: {output_path}")

    # 创建目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 下载
    urllib.request.urlretrieve(url, output_path)
    print(f"下载完成: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="下载 LCCC 数据集")
    parser.add_argument("--version", choices=["base", "large", "all"], default="base",
                        help="数据集版本: base (约350MB), large (约580MB), all")
    args = parser.parse_args()

    versions = ["base", "large"] if args.version == "all" else [args.version]

    for version in versions:
        for filename in FILES[version]:
            url = f"{MIRROR}/{filename}"
            output_path = os.path.join(DATA_DIR, filename)

            if os.path.exists(output_path):
                print(f"文件已存在，跳过: {output_path}")
                continue

            try:
                download_file(url, output_path)
            except Exception as e:
                print(f"下载失败: {e}")
                print("请手动下载或使用其他镜像")


if __name__ == "__main__":
    main()
