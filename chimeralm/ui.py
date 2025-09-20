"""Gradio Web UI for ChimeraLM sequence classification."""

import logging
from pathlib import Path

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import torch

import chimeralm
from chimeralm.data.tokenizer import load_tokenizer_from_hyena_model


class ChimeraLMPredictor:
    """ChimeraLM predictor for web interface."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        """Load the ChimeraLM model and tokenizer."""
        try:
            logging.info("Loading ChimeraLM model...")
            self.model = chimeralm.models.ChimeraLM.from_pretrained("yangliz5/chimeralm")
            self.model.eval()
            self.model.to(self.device)

            self.tokenizer = load_tokenizer_from_hyena_model("hyenadna-small-32k-seqlen")
            logging.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def predict(self, sequence: str) -> tuple[str, float, dict]:
        """Predict if a DNA sequence is chimeric or biological."""
        if not sequence or not sequence.strip():
            return "Please enter a DNA sequence", 0.0, {}

        # Clean and validate sequence
        sequence = sequence.strip().upper()
        # also consider lowercase
        valid_chars = set("ACGTNacgtn")
        if not all(c in valid_chars for c in sequence):
            return "Invalid characters in sequence. Only A, C, G, T, N are allowed.", 0.0, {}

        sequence = sequence.upper()

        try:
            # Tokenize sequence
            tokenized = self.tokenizer(sequence, truncation=True, padding=True, max_length=32768, return_tensors="pt")

            # Extract input_ids and move to device
            input_ids = tokenized["input_ids"].to(self.device)
            input_quals = None  # We don't have quality scores for web input

            # Make prediction
            with torch.no_grad():
                logits = self.model(input_ids, input_quals)
                probabilities = torch.softmax(logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()

            # Interpret results
            class_names = ["Biological", "Chimeric Artifact"]
            prediction = class_names[predicted_class]

            # Create confidence breakdown
            confidence_breakdown = {
                "Biological": f"{probabilities[0][0].item():.3f}",
                "Chimeric Artifact": f"{probabilities[0][1].item():.3f}",
            }

            return prediction, confidence, confidence_breakdown

        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return f"Prediction failed: {str(e)}", 0.0, {}


def create_interface():
    """Create the Gradio interface."""
    predictor = ChimeraLMPredictor()

    def predict_sequence(sequence):
        prediction, confidence, breakdown = predictor.predict(sequence)

        # Format output
        result_text = f"**Prediction:** {prediction}\n**Confidence:** {confidence:.3f}"

        if breakdown:
            result_text += "\n\n**Confidence Breakdown:**\n"
            for class_name, prob in breakdown.items():
                result_text += f"- {class_name}: {prob}\n"

        # Create bar plot
        if breakdown:
            classes = list(breakdown.keys())
            probabilities = [float(prob) for prob in breakdown.values()]

            # Create colors based on prediction
            colors = []
            for i, class_name in enumerate(classes):
                if class_name == prediction:
                    colors.append(
                        "#2E8B57" if prediction == "Biological" else "#DC143C"
                    )  # Green for Biological, Red for Chimeric
                else:
                    colors.append("#D3D3D3")  # Light gray for non-predicted class

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=classes,
                        y=probabilities,
                        marker_color=colors,
                        text=[f"{p:.3f}" for p in probabilities],
                        textposition="auto",
                    )
                ]
            )

            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Classification",
                yaxis_title="Probability",
                yaxis=dict(range=[0, 1]),
                height=400,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            fig.update_traces(textfont_size=14, textfont_color="black")
        else:
            # Create empty plot for error cases
            fig = go.Figure()
            fig.update_layout(
                title="Prediction Probabilities",
                xaxis_title="Classification",
                yaxis_title="Probability",
                height=400,
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

        return result_text, fig

    # Example sequences
    examples = [
        ["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"],
        ["AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"],
        ["ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"],
        ["GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"],
    ]

    with gr.Blocks(title="ChimeraLM - Chimeric Read Detector", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 🧬 ChimeraLM - Chimeric Read Detector
            
            A deep learning model to identify chimeric artifacts introduced by whole genome amplification (WGA).
            
            **Instructions:**
            1. Enter a DNA sequence
            2. Only use standard nucleotides: A, C, G, T, N
            3. The model will classify the sequence as either **Biological** or **Chimeric Artifact**
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                sequence_input = gr.Textbox(
                    label="DNA Sequence",
                    placeholder="Enter your DNA sequence here (e.g., ACGTACGTACGT...)",
                    lines=5,
                    max_lines=10,
                )

                predict_btn = gr.Button("🔬 Analyze Sequence", variant="primary", size="lg")

                gr.Examples(examples=examples, inputs=[sequence_input], label="Example Sequences")

            with gr.Column(scale=1):
                result_output = gr.Markdown(
                    label="Prediction Result", value="Enter a sequence and click 'Analyze Sequence' to see results."
                )

                # Add the plot component
                plot_output = gr.Plot(label="Probability Distribution", value=None)

        gr.Markdown(
            """
            ---
            
            **About ChimeraLM:**
            - Trained to detect chimeric artifacts from whole genome amplification
            - Maximum sequence length: 32,768 nucleotides
            - Model: `yangliz5/chimeralm`
            
            **Citation:**
            ```
            @software{chimeralm2025,
              title={ChimeraLM: A genomic language model to identify chimera artifacts},
              author={Li, Yangyang, Guo, Qingxiang and Yang, Rendong},
              year={2025},
              url={https://github.com/ylab-hi/ChimeraLM}
            }
            ```
            """
        )

        predict_btn.click(fn=predict_sequence, inputs=[sequence_input], outputs=[result_output, plot_output])

    return interface


def main():
    """Launch the Gradio interface."""
    logging.basicConfig(level=logging.INFO)
    interface = create_interface()
    interface.launch(share=False)


if __name__ == "__main__":
    main()
