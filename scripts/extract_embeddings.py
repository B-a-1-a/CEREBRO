import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--encoder", choices=["clip", "dinov2"], default="clip")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
