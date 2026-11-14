"""Single entrypoint: python -m cronlm.cli <command>"""

import argparse
import sys

from . import dataset_gen
from .evaluate import evaluate_checkpoint
from .predict import predict, repl
from .train import run_train


def main(argv=None):
    parser = argparse.ArgumentParser(prog="cronlm", description="CronLM: English -> cron")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen-data", help="generate all splits + golden copy")
    p_gen.add_argument("--out", default="data")
    p_gen.add_argument("--train", type=int, default=500_000)
    p_gen.add_argument("--val", type=int, default=20_000)
    p_gen.add_argument("--test", type=int, default=100_000)
    p_gen.add_argument("--combo", type=int, default=20_000)
    p_gen.add_argument("--paraphrase", type=int, default=10_000)
    p_gen.add_argument("--seed", type=int, default=1337)
    p_gen.add_argument("--golden-src", default="tests/golden.jsonl")

    p_train = sub.add_parser("train", help="train the model")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--online", action="store_true",
                         help="fresh generator samples every step instead of reading train.jsonl")
    p_train.add_argument("--resume", default=None, metavar="PATH")
    p_train.add_argument("--smoke", action="store_true",
                         help="200 steps on 5k examples, under a minute")
    p_train.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p_train.add_argument("--amp", default="none", choices=["none", "fp16", "bf16"])
    p_train.add_argument("--num-workers", type=int, default=0,
                         help="background workers for --online sampling (default 0 = in-process)")

    p_eval = sub.add_parser("eval", help="evaluate a checkpoint")
    p_eval.add_argument("--ckpt", required=True)
    p_eval.add_argument("--split", required=True,
                        choices=["test", "val", "combo", "paraphrase", "golden"])
    p_eval.add_argument("--data-dir", default="data")
    p_eval.add_argument("--batch-size", type=int, default=128)
    p_eval.add_argument("--max-examples", type=int, default=None)

    p_predict = sub.add_parser("predict", help="print the cron for a phrase")
    p_predict.add_argument("--ckpt", required=True)
    p_predict.add_argument("phrase")

    p_repl = sub.add_parser("repl", help="interactive loop")
    p_repl.add_argument("--ckpt", required=True)

    args = parser.parse_args(argv)

    if args.command == "gen-data":
        dataset_gen.build_splits(args.out, train_n=args.train, val_n=args.val,
                                 test_n=args.test, combo_n=args.combo,
                                 paraphrase_n=args.paraphrase, seed=args.seed,
                                 golden_src=args.golden_src)
    elif args.command == "train":
        run_train(args)
    elif args.command == "eval":
        evaluate_checkpoint(args.ckpt, args.split, data_dir=args.data_dir,
                            batch_size=args.batch_size,
                            max_examples=args.max_examples)
    elif args.command == "predict":
        predict(args.ckpt, args.phrase)
    elif args.command == "repl":
        repl(args.ckpt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
