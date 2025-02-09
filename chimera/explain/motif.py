from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


class SequenceAnalyzer:
    """Analyze sequences using state space models."""

    def __init__(self, model, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def get_state_activations(self, sequence: str) -> dict[str, torch.Tensor]:
        """Extract state activations for a sequence."""
        # Convert sequence to model input format
        if self.tokenizer:
            input_ids = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        else:
            # For character-level input
            input_ids = torch.tensor([[ord(c) for c in sequence]], device=self.device)

        activations = {}

        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()

            return hook

        # Register hooks for state space layers
        hooks = []
        for name, module in self.model.named_modules():
            if "state_space" in name:  # Adjust based on your model architecture
                hooks.append(module.register_forward_hook(hook_fn(name)))

        # Forward pass
        with torch.no_grad():
            _output = self.model(input_ids)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations

    def analyze_sequence_importance(self, sequence: str) -> dict[str, np.ndarray]:
        """Analyze importance of each position in sequence."""
        original_pred = self.get_prediction(sequence)
        importance_scores = []

        # Analyze each position
        for i in range(len(sequence)):
            # Create modified sequence with masked position
            mod_seq = sequence[:i] + "N" + sequence[i + 1 :]
            mod_pred = self.get_prediction(mod_seq)

            # Calculate importance as prediction change
            importance = abs(original_pred - mod_pred)
            importance_scores.append(float(importance))

        return {"position_importance": np.array(importance_scores), "sequence": sequence}

    def extract_motifs(self, sequence: str, window_sizes: list[int] = [3, 4, 5]) -> list[dict]:
        """Extract significant motifs from sequence."""
        motifs = []
        activations = self.get_state_activations(sequence)

        # Analyze different window sizes
        for window_size in window_sizes:
            for i in range(len(sequence) - window_size + 1):
                motif = sequence[i : i + window_size]

                # Get average activation for this motif
                motif_activations = {}
                for name, acts in activations.items():
                    if len(acts.shape) >= 2:  # Check if activation has sequence dimension
                        motif_acts = acts[0, i : i + window_size].mean(dim=0)
                        motif_activations[name] = motif_acts.cpu().numpy()

                # Calculate motif significance
                significance = self.calculate_motif_significance(motif_activations)

                if significance > 0.5:  # Adjust threshold as needed
                    motifs.append(
                        {
                            "motif": motif,
                            "position": i,
                            "significance": float(significance),
                            "activations": motif_activations,
                        }
                    )

        return motifs

    def analyze_state_patterns(self, sequences: list[str]) -> dict[str, list]:
        """Analyze how states respond to different sequence patterns."""
        state_patterns = defaultdict(list)

        for seq in sequences:
            activations = self.get_state_activations(seq)

            # Analyze each state space layer
            for name, acts in activations.items():
                if len(acts.shape) >= 2:  # Check if activation has sequence dimension
                    # Get most active states
                    state_activity = acts[0].mean(dim=0)
                    top_states = torch.topk(state_activity, k=5)

                    state_patterns[name].append(
                        {
                            "sequence": seq,
                            "top_states": top_states.indices.cpu().numpy(),
                            "activations": top_states.values.cpu().numpy(),
                        }
                    )

        return dict(state_patterns)

    def visualize_analysis(self, sequence: str):
        """Create comprehensive visualization of sequence analysis."""
        # Get all analyses
        importance = self.analyze_sequence_importance(sequence)
        motifs = self.extract_motifs(sequence)
        activations = self.get_state_activations(sequence)

        # Create visualization
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        # Plot position importance
        axes[0].plot(importance["position_importance"])
        axes[0].set_title("Position Importance")
        axes[0].set_xlabel("Sequence Position")
        axes[0].set_ylabel("Importance Score")

        # Plot motif locations
        motif_scores = np.zeros(len(sequence))
        for motif in motifs:
            pos = motif["position"]
            length = len(motif["motif"])
            motif_scores[pos : pos + length] += motif["significance"]

        axes[1].plot(motif_scores)
        axes[1].set_title("Motif Significance")
        axes[1].set_xlabel("Sequence Position")
        axes[1].set_ylabel("Significance Score")

        # Plot state activations
        for name, acts in activations.items():
            if len(acts.shape) >= 2:
                sns.heatmap(acts[0].cpu().numpy().T, ax=axes[2])
                axes[2].set_title(f"State Activations - {name}")
                axes[2].set_xlabel("Sequence Position")
                axes[2].set_ylabel("State Dimension")
                break  # Only plot first state space layer

        plt.tight_layout()
        return fig

    def get_prediction(self, sequence: str) -> float:
        """Get model prediction for sequence."""
        if self.tokenizer:
            input_ids = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        else:
            input_ids = torch.tensor([[ord(c) for c in sequence]], device=self.device)

        with torch.no_grad():
            output = self.model(input_ids)

        return output.logits[0].softmax(dim=-1)[1].item()  # Assuming binary classification


# Usage example:
def analyze_sequences(model, sequences: list[str]):
    """Analyze a set of sequences and print comprehensive report."""
    analyzer = SequenceAnalyzer(model)

    print("=== Sequence Analysis Report ===")
    for seq in sequences:
        print(f"\nAnalyzing sequence: {seq}")

        # Get importance scores
        importance = analyzer.analyze_sequence_importance(seq)
        print("\nPosition Importance:")
        for pos, score in enumerate(importance["position_importance"]):
            if score > 0.5:  # Adjust threshold as needed
                print(f"Position {pos} ({seq[pos]}): {score:.3f}")

        # Get significant motifs
        motifs = analyzer.extract_motifs(seq)
        print("\nSignificant Motifs:")
        for motif in motifs:
            print(
                f"Motif: {motif['motif']} at position {motif['position']} (significance: {motif['significance']:.3f})"
            )

        # Visualize analysis
        fig = analyzer.visualize_analysis(seq)
        plt.show()

        # Prediction
        pred = analyzer.get_prediction(seq)
        print(f"\nModel prediction: {pred:.3f}")

        print("\n" + "=" * 50)
