import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


class Mamba2Analyzer:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def extract_state_activations(self, sequence: str) -> dict[str, torch.Tensor]:
        """Extract state activations from Mamba2 layers."""
        # Convert sequence to model input format
        input_seq = torch.tensor([[ord(c) for c in sequence]], device=self.device)

        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                # Capture both state activations and selective parameters
                if isinstance(output, tuple):
                    activations[f"{name}_state"] = output[0].detach()
                    if len(output) > 1:  # If selective parameters are available
                        activations[f"{name}_delta"] = output[1].detach()
                else:
                    activations[name] = output.detach()

            return hook

        # Register hooks for Mamba blocks
        hooks = []
        for name, module in self.model.named_modules():
            if "mamba_block" in name:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        with torch.no_grad():
            output = self.model(input_seq)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations, output

    def analyze_selective_mechanism(self, sequence: str) -> dict[str, np.ndarray]:
        """Analyze the selective mechanism (∆ and B parameters)."""
        activations, _ = self.extract_state_activations(sequence)

        selective_patterns = {}
        for name, activation in activations.items():
            if "delta" in name:
                # Analyze how selective mechanism responds to different parts of sequence
                delta_values = activation.cpu().numpy()
                selective_patterns[name] = {
                    "mean_delta": delta_values.mean(axis=-1),
                    "max_delta": delta_values.max(axis=-1),
                    "pattern_strength": np.abs(delta_values).mean(axis=-1),
                }

        return selective_patterns

    def get_position_importance(self, sequence: str) -> np.ndarray:
        """Calculate importance of each position using state activations."""
        original_activations, original_output = self.extract_state_activations(sequence)
        original_pred = original_output.softmax(dim=-1)[0, 1].item()

        importance_scores = []

        # Analyze each position
        for i in range(len(sequence)):
            # Create modified sequence
            mod_seq = sequence[:i] + "N" + sequence[i + 1 :]
            mod_activations, mod_output = self.extract_state_activations(mod_seq)
            mod_pred = mod_output.softmax(dim=-1)[0, 1].item()

            # Calculate importance as prediction change
            importance = abs(original_pred - mod_pred)
            importance_scores.append(importance)

        return np.array(importance_scores)

    def find_sequence_patterns(self, sequence: str, window_sizes=None) -> list[dict]:
        """Find significant sequence patterns using state activations."""
        if window_sizes is None:
            window_sizes = [3, 4, 5]
        activations, _ = self.extract_state_activations(sequence)
        patterns = []

        # Analyze state activations for patterns
        for name, activation in activations.items():
            if "_state" in name:
                state_acts = activation.cpu().numpy()[0]  # Remove batch dimension

                # Look for patterns in different window sizes
                for window_size in window_sizes:
                    for i in range(len(sequence) - window_size + 1):
                        # Get subsequence and its activations
                        subseq = sequence[i : i + window_size]
                        subseq_acts = state_acts[i : i + window_size]

                        # Calculate pattern significance
                        pattern_strength = np.mean(np.abs(subseq_acts))
                        if pattern_strength > np.mean(np.abs(state_acts)):
                            patterns.append(
                                {"sequence": subseq, "position": i, "strength": float(pattern_strength), "layer": name}
                            )

        return patterns

    def visualize_analysis(self, sequence: str):
        """Create comprehensive visualization of sequence analysis."""
        # Get all analyses
        activations, _ = self.extract_state_activations(sequence)
        importance_scores = self.get_position_importance(sequence)
        selective_patterns = self.analyze_selective_mechanism(sequence)
        self.find_sequence_patterns(sequence)

        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Plot position importance
        axes[0].plot(importance_scores)
        axes[0].set_title("Position Importance")
        axes[0].set_xlabel("Sequence Position")
        axes[0].set_ylabel("Importance Score")

        # Plot selective mechanism patterns
        for name, patterns in selective_patterns.items():
            axes[1].plot(patterns["pattern_strength"], label=name)
        axes[1].set_title("Selective Mechanism Patterns")
        axes[1].set_xlabel("Sequence Position")
        axes[1].set_ylabel("Pattern Strength")
        axes[1].legend()

        # Plot state activations
        for name, activation in activations.items():
            if "_state" in name:
                sns.heatmap(activation[0].cpu().numpy().T, ax=axes[2])
                axes[2].set_title(f"State Activations - {name}")
                axes[2].set_xlabel("Sequence Position")
                axes[2].set_ylabel("State Dimension")
                break  # Only plot first state space layer

        plt.tight_layout()
        return fig


def analyze_mamba2_sequence(model, sequence: str):
    """Run comprehensive analysis on a sequence."""
    analyzer = Mamba2Analyzer(model)

    # Get importance scores
    importance = analyzer.get_position_importance(sequence)
    for _pos, score in enumerate(importance):
        if score > 0.1:  # Adjust threshold as needed
            pass

    # Get sequence patterns
    patterns = analyzer.find_sequence_patterns(sequence)
    for _pattern in patterns:
        pass

    # Analyze selective mechanism
    selective = analyzer.analyze_selective_mechanism(sequence)
    for _name, patterns in selective.items():
        pass

    # Visualize analysis
    analyzer.visualize_analysis(sequence)
    plt.show()


# Example usage:
"""
# Initialize with your trained model
model = your_trained_mamba2_model
sequence = "ATCGGTCGATCG"
analyze_mamba2_sequence(model, sequence)
"""
