#!/bin/bash
# toy_submit_all.sh
# Kills job THE MOMENT epoch 1 finishes → best possible resume test

set -e

EXP_NAME="perfect_resume_test_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="slurm_logs"
CKPT_DIR="./checkpoints/${EXP_NAME}"
mkdir -p $LOG_DIR

echo "=========================================="
echo "PERFECT RESUME TEST"
echo "=========================================="
echo "→ Will kill job right after 'Epoch 1' completes"
echo "→ Guarantees epoch_0.pth + epoch_1.pth exist"
echo "→ Resume will start from epoch 2 → epoch_2.pth appears = 100% proof"
echo ""
echo "Experiment: $EXP_NAME"
echo "Checkpoint: $CKPT_DIR"
echo "=========================================="

# Create temporary sbatch script
SBATCH_SCRIPT=$(mktemp)
cat > "$SBATCH_SCRIPT" <<'ENDSCRIPT'
#!/bin/bash
#SBATCH --job-name=JOBNAME
#SBATCH --output=LOGDIR/JOBNAME.out
#SBATCH --error=LOGDIR/JOBNAME.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-node=2
#SBATCH --mem=100G
#SBATCH --partition=gpu
#SBATCH --account=gpu
#SBATCH --exclude=scholar-g[000-003]

echo "=========================================="
echo "Started at $(date)"
echo "Experiment: JOBNAME"
echo "Checkpoint: CKPTDIR"
echo "=========================================="

python3 PYTHONPATH/my_whisper/train/main.py \
    --config configs/base_lora.yaml \
    TSV_PATH "./logs/JOBNAME_log.tsv" \
    CHECKPOINT_DIR "CKPTDIR" \
    EXP_NAME "JOBNAME" \
    MODEL.BASE "small.en" \
    LORA.RANK "64" \
    LORA.ALPHA "128" \
    LORA.DROPOUT "0.1" \
    LORA.TARGET_BLOCKS "0-23" \
    TRAIN_DATASETS.0.name "commonvoice" \
    TRAIN_DATASETS.0.split "train[:30000]" \
    VALIDATION_DATASETS.0.name "commonvoice" \
    VALIDATION_DATASETS.0.split "train[30000:35000]" \
    TRAINER.BATCH_SIZE "8" \
    TRAINER.ACCUMULATION_STEPS "4" \
    TRAINER.MAX_EPOCHS "8" \
    TRAINER.EARLY_STOPPING_PATIENCE "3" \
    TRAINER.LR "1e-5"

EXITCODE=$?
echo ""
echo "=========================================="
echo "Training finished with exit code: $EXITCODE"
echo "=========================================="
ENDSCRIPT

# Replace placeholders
sed -i "s|JOBNAME|${EXP_NAME}|g" "$SBATCH_SCRIPT"
sed -i "s|LOGDIR|${LOG_DIR}|g" "$SBATCH_SCRIPT"
sed -i "s|CKPTDIR|${CKPT_DIR}|g" "$SBATCH_SCRIPT"
sed -i "s|PYTHONPATH|/scratch/scholar/$USER/Whisper|g" "$SBATCH_SCRIPT"

# Submit the job
JOB_ID=$(sbatch --parsable "$SBATCH_SCRIPT")
rm "$SBATCH_SCRIPT"

echo ""
echo "Job submitted: ID = $JOB_ID"
echo "Log: ${LOG_DIR}/${EXP_NAME}.out"
echo ""
