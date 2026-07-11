import pandas as pd
import json

class ExperimentTracker:
    """Manages the history of experiments and exports them."""

    def __init__(self):
        self.history = []
        self._id_counter = 1
        
    def add_experiment(self, pipeline_config: list, baseline_metrics: dict, experiment_metrics: dict):
        """Adds a new experiment record to the history."""
        config_str = " -> ".join([step['step'] for step in pipeline_config]) if pipeline_config else "Original (No Processing)"
        
        # Calculate deltas if eval mode
        if experiment_metrics.get("mode") == "eval" and baseline_metrics.get("mode") == "eval":
            cer_delta = experiment_metrics['cer'] - baseline_metrics['cer']
            wer_delta = experiment_metrics['wer'] - baseline_metrics['wer']
            cer_str = f"{experiment_metrics['cer']:.2f}% ({cer_delta:+.2f}%)"
            wer_str = f"{experiment_metrics['wer']:.2f}% ({wer_delta:+.2f}%)"
        else:
            cer_str = "N/A"
            wer_str = "N/A"
            
        lat_delta = experiment_metrics['latency'] - baseline_metrics['latency']
        conf_delta = experiment_metrics['confidence'] - baseline_metrics['confidence']
        
        record = {
            "ID": f"EXP-{self._id_counter:03d}",
            "Pipeline": config_str,
            "CER": cer_str,
            "WER": wer_str,
            "Latency": f"{experiment_metrics['latency']:.2f}s ({lat_delta:+.2f}s)",
            "Confidence": f"{experiment_metrics['confidence']:.2f} ({conf_delta:+.2f})",
            "Raw_Config": json.dumps(pipeline_config)
        }
        
        self.history.append(record)
        self._id_counter += 1
        
    def get_history_df(self) -> pd.DataFrame:
        """Returns the history as a Pandas DataFrame for Streamlit."""
        if not self.history:
            return pd.DataFrame(columns=["ID", "Pipeline", "CER", "WER", "Latency", "Confidence"])
        
        # We don't want to show Raw_Config in the main table to save space
        df = pd.DataFrame(self.history)
        return df.drop(columns=["Raw_Config"])
