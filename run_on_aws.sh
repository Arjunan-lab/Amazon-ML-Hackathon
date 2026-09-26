#!/usr/bin/env bash
# =============================================================================
# Amazon ML Challenge: Business Entity Resolution
# One-Click End-to-End Pipeline for AWS Cloud GPU Instances (e.g. g5.2xlarge)
# =============================================================================

set -e
export PYTHONUNBUFFERED=1

echo "=================================================================="
echo "AMAZON ML CHALLENGE: LAUNCHING AWS PIPELINE"
echo "=================================================================="

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 1. Check Hardware
echo "--> Checking Hardware Status..."
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "Running in high-speed multi-core CPU mode (OpenBLAS + RapidFuzz)."
fi



# 3. Create required output directories
mkdir -p output
mkdir -p models

# 4. Run end-to-end training and inference
echo "--> Executing End-to-End Resolution Pipeline..."
python -m code.business_entity_resolution.src.pipeline \
    --train-dir data/train \
    --test-dir data/test \
    --output-dir output \
    --top-k 15 \
    --max-train-samples 40000

# 5. Validate outputs against official competition rules
echo "--> Validating Submission Files..."
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/test

# 6. Package final submission zip
echo "--> Packaging Final Submission Archive..."
python utils/package_submission.py \
    --team-name "Team_Amazon_ER" \
    --test-dir data/test

echo "=================================================================="
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "Download 'Team_Amazon_ER_submission.zip' and upload matching_results.tsv to the portal."
echo "=================================================================="
