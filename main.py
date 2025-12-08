# -*- coding: utf-8 -*-
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True,
                        choices=["preprocess","train_cnn","eval","gui"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_path", type=str, default="models/emotion_resnet.pt")
    args = parser.parse_args()

    if args.mode == "preprocess":
        from scripts.preprocess import split_and_save
        split_and_save(data_dir=args.data_dir)

    elif args.mode == "train_cnn":
        from scripts.train_resnet import main as train_main
        train_main(data_dir=args.data_dir, model_path=args.model_path,
                   epochs=args.epochs, lr=args.lr, batch_size=args.batch)

    elif args.mode == "eval":
        from scripts.evaluate import evaluate as eval_main
        eval_main(data_dir=args.data_dir, model_path=args.model_path)

    elif args.mode == "gui":
        if not os.path.exists(args.model_path):
            print(f"Model not found at {args.model_path}. Run train_cnn first.")
            return
        from app.gui import main as gui_main
        gui_main(model_path=args.model_path)

if __name__ == "__main__":
    main()
