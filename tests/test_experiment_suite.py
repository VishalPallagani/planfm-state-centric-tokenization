import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))


class TestTokenizerStudyExperimentSuite(unittest.TestCase):
    def setUp(self):
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"xgboost unavailable: {exc}")

        self.original_data_dir = PROJECT_ROOT / "data"
        self.test_root = Path(tempfile.mkdtemp(prefix="tokenizer_study_suite_"))
        self.addCleanup(lambda: shutil.rmtree(self.test_root, ignore_errors=True))

        self.test_data_dir = self.test_root / "data"
        self.output_root = self.test_root / "outputs"
        self.domains = ["blocks", "gripper"]
        self.tokenizers = ["random", "simhash"]

        self._create_minimal_dataset()

    def _create_minimal_dataset(self):
        pddl_dir = self.test_data_dir / "pddl"
        states_dir = self.test_data_dir / "states"
        pddl_dir.mkdir(parents=True)
        states_dir.mkdir(parents=True)

        splits = ["train", "validation", "test-interpolation", "test-extrapolation"]
        for domain in self.domains:
            src_domain_pddl_dir = self.original_data_dir / "pddl" / domain
            src_domain_states_dir = self.original_data_dir / "states" / domain

            dst_domain_pddl_dir = pddl_dir / domain
            dst_domain_states_dir = states_dir / domain
            dst_domain_pddl_dir.mkdir()
            dst_domain_states_dir.mkdir()

            shutil.copy(
                src_domain_pddl_dir / "domain.pddl",
                dst_domain_pddl_dir / "domain.pddl",
            )

            for split in splits:
                src_split_pddl = src_domain_pddl_dir / split
                src_split_states = src_domain_states_dir / split
                dst_split_pddl = dst_domain_pddl_dir / split
                dst_split_states = dst_domain_states_dir / split
                dst_split_pddl.mkdir()
                dst_split_states.mkdir()

                traj_files = sorted(src_split_states.glob("*.traj")) if src_split_states.exists() else []
                if not traj_files:
                    continue

                traj_file = traj_files[0]
                prob_name = traj_file.stem
                pddl_file = src_split_pddl / f"{prob_name}.pddl"

                shutil.copy(traj_file, dst_split_states / traj_file.name)
                if pddl_file.exists():
                    shutil.copy(pddl_file, dst_split_pddl / pddl_file.name)

    def test_study_runner_reduced(self):
        cmd = [
            sys.executable,
            "-m",
            "code.experiments.run_tokenizer_study",
            "--source_data_dir",
            str(self.test_data_dir),
            "--output_root",
            str(self.output_root),
            "--study_name",
            "reduced_suite",
            "--overwrite",
            "--tokenizers",
            *self.tokenizers,
            "--domains",
            *self.domains,
            "--models",
            "xgboost",
            "--modes",
            "state",
            "--seeds",
            "13",
            "--device",
            "cpu",
            "--num_workers",
            "0",
            "--xgb_n_jobs",
            "1",
            "--xgb_n_estimators",
            "4",
            "--xgb_max_depth",
            "2",
            "--xgb_early_stopping",
            "2",
            "--skip_validation",
            "--max_problems",
            "1",
            "--analysis_seed",
            "7",
        ]

        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != 0:
            self.fail(f"Study runner failed.\n{result.stdout}")

        run_root = self.output_root / "reduced_suite"
        self.assertTrue((run_root / "manifest.json").exists())
        self.assertTrue((run_root / "analysis" / "study_report.md").exists())
        self.assertTrue((run_root / "analysis" / "best_config_by_regime_model_mode.csv").exists())
        self.assertTrue((run_root / "logs" / "commands.log").exists())


if __name__ == "__main__":
    unittest.main()
