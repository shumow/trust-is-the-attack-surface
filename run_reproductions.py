"""Run the reproducibility demos from one entry point.

Default mode runs the central demos used in the README. The full mode runs every
demo script; it is slower, but useful before publishing regenerated figures.
"""
import argparse
import subprocess
import sys


CORE_DEMOS = [
    "demo_conservation.py",
    "demo_contagion.py",
    "demo_contagion_ppm.py",
    "demo_real_text_cache.py",
]

FULL_DEMOS = [
    "demo_conservation.py",
    "demo_contagion.py",
    "demo_contagion_orders.py",
    "demo_contagion_divis.py",
    "demo_contagion_regimes.py",
    "demo_contagion_hijack.py",
    "demo_contagion_hijack_bb.py",
    "demo_contagion_hijack_quine.py",
    "demo_contagion_worm.py",
    "demo_contagion_joint.py",
    "demo_contagion_ppm.py",
    "demo_real_text_cache.py",
    "demo_sensitivity.py",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all demos instead of the central reproducibility set",
    )
    args = parser.parse_args()

    for script in FULL_DEMOS if args.full else CORE_DEMOS:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
