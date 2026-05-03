import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
