import os
import sys

from hyde_v2_core import HyDEDemoV2


def main():
    if len(sys.argv) > 1:
        corpus_path = sys.argv[1]
    else:
        corpus_path = "data/corpus.jsonl"

    if not os.path.exists(corpus_path):
        print(f"Error: Corpus file not found: {corpus_path}")
        sys.exit(1)

    demo = HyDEDemoV2(corpus_path)
    demo.run_cli()


if __name__ == "__main__":
    main()