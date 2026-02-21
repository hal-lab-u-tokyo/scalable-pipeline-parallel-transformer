import os
import argparse
from torchvision import datasets
#python prepare_dataset.py --outdir ./dataset --dataset cifar-10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True, help="保存先のルートディレクトリ (例: /work/group/user/dataset)")
    parser.add_argument("--dataset", type=str, default="cifar-10", choices=["cifar-10", "cifar-100"], help="ダウンロードするデータセット")
    args = parser.parse_args()
    
    # dataset_factory.py のロジックに合わせてサブディレクトリを作成
    # factory側で root = os.path.join(root, "cifar-10") としているため
    save_root = os.path.join(args.outdir, args.dataset)
    
    os.makedirs(save_root, exist_ok=True)
    print(f"Downloading {args.dataset} to: {save_root}")

    if args.dataset == "cifar-10":
        # train=True/False 両方ダウンロードして展開しておく
        datasets.CIFAR10(root=save_root, train=True, download=True)
        datasets.CIFAR10(root=save_root, train=False, download=True)
        
    elif args.dataset == "cifar-100":
        datasets.CIFAR100(root=save_root, train=True, download=True)
        datasets.CIFAR100(root=save_root, train=False, download=True)

    print("Done! Download complete.")

if __name__ == "__main__":
    main()