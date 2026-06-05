"""
run_pipeline.py
===============
End-to-end runner for the hallucination research pipeline.

Stages:
  1. Build corpora
  2. Compute data quality metrics
  3. Compute hallucination index
  4. Correlation analysis + figures

Usage:
    python run_pipeline.py --full          # run all stages
    python run_pipeline.py --stage 1       # build corpora only
    python run_pipeline.py --stage 1 2 3   # build + score + evaluate
"""

import argparse
import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_stage(stage_num: int, label: str, fn):
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"STAGE {stage_num}: {label}")
    logger.info("=" * 70)
    t0 = time.time()
    try:
        fn()
        logger.info(f"  Stage {stage_num} completed in {time.time() - t0:.1f}s")
    except Exception as e:
        logger.error(f"  Stage {stage_num} FAILED: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Hallucination Research Pipeline"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run all pipeline stages"
    )
    parser.add_argument(
        "--stage", nargs="+", type=int, choices=[1, 2, 3, 4],
        help="Run specific stage(s): 1=corpora, 2=quality, 3=HI, 4=analysis"
    )
    args = parser.parse_args()

    if not args.full and not args.stage:
        parser.print_help()
        sys.exit(1)

    stages_to_run = list(range(1, 5)) if args.full else sorted(set(args.stage))

    # Import lazily to avoid slow startup when only partial stages needed
    stage_map = {
        1: ("Build Experimental Corpora",       _stage1),
        2: ("Compute Data Quality Metrics",      _stage2),
        3: ("Compute Hallucination Index (HI)",  _stage3),
        4: ("Correlation Analysis & Figures",    _stage4),
    }

    t_total = time.time()
    for s in stages_to_run:
        label, fn = stage_map[s]
        run_stage(s, label, fn)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"PIPELINE COMPLETE — total time: {time.time() - t_total:.1f}s")
    logger.info("Results saved to: results/")
    logger.info("=" * 70)


# ── Stage functions ───────────────────────────────────────────────────────────

def _stage1():
    from src.data_quality.corpus_builder import build_all_corpora
    build_all_corpora()


def _stage2():
    from src.data_quality.quality_metrics import run
    run()


def _stage3():
    from src.evaluation.hallucination_index import run
    run()


def _stage4():
    from src.analysis.correlation import run
    run()


if __name__ == "__main__":
    main()
