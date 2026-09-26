#!/usr/bin/env python3
"""Submission Packaging and Verification Utility (Stage 6).

Automates the creation of the official competition submission zip:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md

Performs automated local validation using utils/validate_submission.py before packaging.
"""

import argparse
import os
import subprocess
import sys
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(description="Package submission zip for Amazon ML Challenge")
    parser.add_argument("--team-name", default="Team_Amazon_ER", help="Your team name for the submission zip file")
    parser.add_argument("--test-dir", default="data/test", help="Path to test dataset directory")
    parser.add_argument("--skip-validation", action="store_true", help="Skip running validate_submission.py")
    return parser.parse_args()


def validate_outputs(matching_file: str, candidate_file: str, test_dir: str):
    print("=" * 70)
    print("STEP 1: VALIDATING SUBMISSION FILES AGAINST OFFICIAL COMPETITION RULES")
    print("=" * 70)

    val_script = "utils/validate_submission.py"
    if not os.path.exists(val_script):
        print(f"Error: {val_script} not found.")
        sys.exit(1)

    cmd = [
        sys.executable,
        val_script,
        "--matching", matching_file,
        "--candidate", candidate_file,
        "--test-dir", test_dir,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        print("\n[!] CRITICAL: Validation failed. Fix the issues above before submitting.")
        sys.exit(1)

    print("--> All outputs successfully passed validation!\n")


def build_submission_zip(team_name: str):
    print("=" * 70)
    print("STEP 2: CREATING OFFICIAL SUBMISSION ZIP PACKAGE")
    print("=" * 70)

    zip_filename = f"{team_name}_submission.zip"
    required_files = [
        ("output/matching_results.tsv", "output/matching_results.tsv"),
        ("output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        ("Documentation_template.md", "Documentation_template.md"),
        ("code/business_entity_resolution/README.md", "code/business_entity_resolution/README.md"),
        ("code/business_entity_resolution/requirements.txt", "code/business_entity_resolution/requirements.txt"),
    ]

    # Verify presence
    missing_files = [src for src, _ in required_files if not os.path.exists(src)]
    if missing_files:
        print("[!] Cannot package submission zip: The following required files are missing:")
        for mf in missing_files:
            print(f"    - {mf}")
        print("\n--> Run the end-to-end pipeline first to generate the output files:")
        print("    poetry run python -m code.business_entity_resolution.src.pipeline")
        sys.exit(1)

    # Source code directory
    src_dir = "code/business_entity_resolution/src"
    if not os.path.exists(src_dir):
        print(f"Error: Source code directory not found: {src_dir}")
        sys.exit(1)

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add required top-level files
        for src_path, arcname in required_files:
            zipf.write(src_path, arcname)
            print(f"  Added: {arcname}")

        # Add all source code files
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith((".py", ".md", ".txt")) and not file.startswith("."):
                    full_p = os.path.join(root, file)
                    arc_p = os.path.relpath(full_p, ".")
                    zipf.write(full_p, arc_p)
                    print(f"  Added: {arc_p}")

    zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
    print("\n" + "=" * 70)
    print(f"SUCCESS: Submission zip created at '{zip_filename}' ({zip_size_mb:.2f} MB)")
    print("Ready for upload to the competition portal!")
    print("=" * 70)


def main():
    args = parse_args()
    matching_file = "output/matching_results.tsv"
    candidate_file = "output/candidate_pairs.tsv"

    if not args.skip_validation:
        validate_outputs(matching_file, candidate_file, args.test_dir)

    build_submission_zip(args.team_name)


if __name__ == "__main__":
    main()
