"""
Tests for evaluate_pipeline orchestrator.
"""

# pylint: disable=protected-access,attribute-defined-outside-init

from pathlib import Path
from unittest.mock import Mock, mock_open, patch

from fusion.sim.evaluate_pipeline import (EvaluationPipeline,
                                          run_evaluation_pipeline)


class TestEvaluationPipeline:
    """Test cases for EvaluationPipeline class."""

    def setup_method(self):
        """Set up test configuration."""
        self.test_config = {
            "model_evaluation": {
                "model_path": "/path/to/model",
                "test_configs": [
                    {"network": "NSFNet", "erlang": "300"},
                    {"network": "NSFNet", "erlang": "600"},
                ],
            }
        }

    def test_initialization(self):
        """Test EvaluationPipeline initialization."""
        pipeline = EvaluationPipeline(self.test_config)
        assert pipeline.config == self.test_config
        assert pipeline.results == {}
        assert pipeline.model_manager is not None

    @patch("fusion.sim.evaluate_pipeline.ModelManager")
    @patch("fusion.sim.evaluate_pipeline.BatchRunner")
    @patch("fusion.sim.evaluate_pipeline.log_message")
    def test_evaluate_model(self, _mock_log, mock_runner_class, mock_manager_class):
        """Test model evaluation."""
        # Mock model manager
        mock_manager = Mock()
        mock_manager.load_model.return_value = None  # Our implementation returns None
        mock_manager_class.return_value = mock_manager

        # Mock batch runner
        mock_runner = Mock()
        mock_runner.run.return_value = [
            {"erlang": 300.0, "stats": {"blocking_probability": 0.01}}
        ]
        mock_runner_class.return_value = mock_runner

        pipeline = EvaluationPipeline(self.test_config)
        pipeline.model_manager = mock_manager

        test_configs = [{"network": "NSFNet", "erlang": "300"}]
        results = pipeline.evaluate_model("/path/to/model", test_configs)

        assert results["model_path"] == "/path/to/model"
        assert len(results["test_results"]) == 1
        assert results["test_results"][0]["config"]["ml_model"] == "placeholder_model"
        assert results["test_results"][0]["config"]["ml_inference"] is True

    @patch("fusion.sim.evaluate_pipeline.BatchRunner")
    @patch("fusion.sim.evaluate_pipeline.log_message")
    def test_compare_algorithms(self, _mock_log, mock_runner_class):
        """Test algorithm comparison."""
        # Mock batch runner
        mock_runner = Mock()
        mock_runner.run.side_effect = [
            [
                {
                    "erlang": 300.0,
                    "stats": {"blocking_probability": 0.01, "total_requests": 1000},
                }
            ],
            [
                {
                    "erlang": 300.0,
                    "stats": {"blocking_probability": 0.02, "total_requests": 1000},
                }
            ],
        ]
        mock_runner_class.return_value = mock_runner

        pipeline = EvaluationPipeline(self.test_config)

        algorithms = {
            "algo1": {"routing_algorithm": "shortest_path"},
            "algo2": {"routing_algorithm": "k_shortest_path"},
        }
        test_scenario = {"network": "NSFNet", "erlang": "300"}

        results = pipeline.compare_algorithms(algorithms, test_scenario)

        assert "algorithms" in results
        assert "algo1" in results["algorithms"]
        assert "algo2" in results["algorithms"]
        assert "metrics" in results
        assert "rankings" in results

    def test_extract_metrics(self):
        """Test metric extraction."""
        pipeline = EvaluationPipeline(self.test_config)

        sim_results = [
            {
                "erlang": 300.0,
                "stats": {
                    "blocking_probability": 0.01,
                    "link_utilization": 0.75,
                    "total_requests": 1000,
                },
            },
            {
                "erlang": 600.0,
                "stats": {
                    "blocking_probability": 0.05,
                    "link_utilization": 0.85,
                    "total_requests": 2000,
                },
            },
        ]

        metrics = pipeline._extract_metrics(sim_results)

        assert metrics["blocking_300.0"] == 0.01
        assert metrics["blocking_600.0"] == 0.05
        assert metrics["utilization_300.0"] == 0.75
        assert metrics["requests_300.0"] == 1000

    def test_generate_comparison_report(self):
        """Test comparison report generation."""
        pipeline = EvaluationPipeline(self.test_config)

        results = {
            "algo1": {"blocking_300": 0.01, "utilization_300": 0.75},
            "algo2": {"blocking_300": 0.02, "utilization_300": 0.70},
        }

        report = pipeline._generate_comparison_report(results)

        assert "algorithms" in report
        assert report["algorithms"] == ["algo1", "algo2"]
        assert "metrics" in report
        assert "rankings" in report
        assert report["rankings"]["blocking_300"] == ["algo1", "algo2"]
        assert report["rankings"]["utilization_300"] == ["algo1", "algo2"]

    @patch("fusion.sim.evaluate_pipeline.ModelManager")
    @patch("fusion.sim.evaluate_pipeline.log_message")
    def test_evaluate_rl_agent(self, _mock_log, mock_manager_class):
        """Test RL agent evaluation."""
        # Mock RL manager
        mock_manager = Mock()
        mock_agent = Mock()
        mock_agent.act.return_value = 0
        mock_manager.load_agent.return_value = mock_agent
        mock_manager_class.return_value = mock_manager

        pipeline = EvaluationPipeline(self.test_config)
        results = pipeline.evaluate_rl_agent("/path/to/agent", episodes=10)

        assert "mean_reward" in results
        assert "std_reward" in results
        assert "mean_blocking" in results
        assert results["episodes"] == 10

    @patch("builtins.open", new_callable=mock_open)
    @patch("fusion.sim.evaluate_pipeline.create_directory")
    @patch("fusion.sim.evaluate_pipeline.log_message")
    def test_generate_markdown_summary(
        self, _mock_log, _mock_create_directory, mock_file
    ):
        """Test markdown summary generation."""
        pipeline = EvaluationPipeline(self.test_config)

        results = {
            "model": {"model_path": "/path/to/model", "test_results": [{}, {}]},
            "comparison": {
                "algorithms": ["algo1", "algo2"],
                "rankings": {
                    "blocking_300": ["algo1", "algo2"],
                    "utilization_300": ["algo2", "algo1"],
                },
            },
            "rl": {
                "episodes": 100,
                "mean_reward": 0.5,
                "std_reward": 0.1,
                "mean_blocking": 0.01,
                "std_blocking": 0.001,
            },
        }

        output_path = Path("/test/output")
        pipeline._generate_markdown_summary(results, output_path)

        mock_file.assert_called_once_with(
            output_path / "summary.md", "w", encoding="utf-8"
        )
        handle = mock_file()

        # Check that key sections were written
        written_content = "".join(call[0][0] for call in handle.write.call_args_list)
        assert "# Evaluation Report Summary" in written_content
        assert "## Model Evaluation" in written_content
        assert "## Algorithm Comparison" in written_content
        assert "## RL Agent Evaluation" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("fusion.sim.evaluate_pipeline.create_directory")
    @patch("fusion.sim.evaluate_pipeline.log_message")
    def test_generate_report(self, _mock_log, mock_create_directory, mock_file):
        """Test complete report generation."""
        pipeline = EvaluationPipeline(self.test_config)

        results = {
            "test_results": [{"config": {}, "results": []}],
            "comparison": {"algorithms": ["algo1"]},
        }

        output_dir = "/test/output"
        pipeline.generate_report(results, output_dir)

        mock_create_directory.assert_called_once_with(output_dir)

        # Check JSON was saved
        assert any(
            "evaluation_results.json" in str(call) for call in mock_file.call_args_list
        )

    @patch("fusion.sim.evaluate_pipeline.EvaluationPipeline.evaluate_model")
    @patch("fusion.sim.evaluate_pipeline.EvaluationPipeline.compare_algorithms")
    @patch("fusion.sim.evaluate_pipeline.EvaluationPipeline.evaluate_rl_agent")
    @patch("fusion.sim.evaluate_pipeline.EvaluationPipeline.generate_report")
    def test_run_full_evaluation(self, mock_report, mock_rl, mock_compare, mock_model):
        """Test full evaluation pipeline."""
        mock_model.return_value = {"model": "results"}
        mock_compare.return_value = {"comparison": "results"}
        mock_rl.return_value = {"rl": "results"}

        pipeline = EvaluationPipeline(self.test_config)

        eval_config = {
            "model_evaluation": {"model_path": "/model", "test_configs": [{}]},
            "algorithm_comparison": {"algorithms": {"algo1": {}}, "scenario": {}},
            "rl_evaluation": {"agent_path": "/agent", "episodes": 50},
            "generate_report": True,
            "output_dir": "/output",
        }

        results = pipeline.run_full_evaluation(eval_config)

        assert "model" in results
        assert "comparison" in results
        assert "rl" in results

        mock_model.assert_called_once()
        mock_compare.assert_called_once()
        mock_rl.assert_called_once()
        mock_report.assert_called_once()

    @patch("fusion.sim.evaluate_pipeline.EvaluationPipeline")
    def test_run_evaluation_pipeline_function(self, mock_pipeline_class):
        """Test convenience function."""
        mock_pipeline = Mock()
        mock_pipeline.run_full_evaluation.return_value = {"test": "results"}
        mock_pipeline_class.return_value = mock_pipeline

        config = {"test": "config"}
        results = run_evaluation_pipeline(config)

        mock_pipeline_class.assert_called_once_with(config)
        mock_pipeline.run_full_evaluation.assert_called_once_with(config)
        assert results == {"test": "results"}
