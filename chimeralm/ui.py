"""Gradio Web UI for ChimeraLM sequence classification."""

import logging

import gradio as gr
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
            return f"Prediction failed: {e}", 0.0, {}


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

            # Create light theme colors based on prediction
            colors = []
            for _i, class_name in enumerate(classes):
                if class_name == prediction:
                    colors.append(
                        "#4CAF50" if prediction == "Biological" else "#F44336"
                    )  # Light green for Biological, Light red for Chimeric
                else:
                    colors.append("#E0E0E0")  # Light gray for non-predicted class

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
                title={
                    "text": "🎯 Prediction Confidence",
                    "font": {"size": 20, "color": "#424242", "family": "Arial, sans-serif"},
                    "x": 0.5,
                    "xanchor": "center",
                },
                xaxis={
                    "title": {"text": "Classification", "font": {"size": 14, "color": "#616161"}},
                    "tickfont": {"size": 12, "color": "#424242"},
                    "gridcolor": "rgba(0,0,0,0.05)",
                    "linecolor": "rgba(0,0,0,0.1)",
                    "showgrid": True,
                    "zeroline": False,
                },
                yaxis={
                    "title": {"text": "Probability", "font": {"size": 14, "color": "#616161"}},
                    "tickfont": {"size": 12, "color": "#424242"},
                    "range": [0, 1.1],
                    "gridcolor": "rgba(0,0,0,0.05)",
                    "linecolor": "rgba(0,0,0,0.1)",
                    "showgrid": True,
                    "zeroline": True,
                    "zerolinecolor": "rgba(0,0,0,0.1)",
                },
                height=450,
                showlegend=False,
                plot_bgcolor="rgba(255,255,255,1)",
                paper_bgcolor="rgba(255,255,255,1)",
                margin={"l": 60, "r": 60, "t": 80, "b": 60},
                font={"family": "Arial, sans-serif"},
            )

            fig.update_traces(
                textfont_size=16,
                textfont_color="white",
                textfont_family="Arial, sans-serif",
                marker_line={"width": 1, "color": "rgba(255,255,255,0.8)"},
                width=0.6,
                opacity=0.9,
            )
        else:
            # Create empty plot for error cases
            fig = go.Figure()
            fig.update_layout(
                title={
                    "text": "🎯 Prediction Confidence",
                    "font": {"size": 20, "color": "#424242", "family": "Arial, sans-serif"},
                    "x": 0.5,
                    "xanchor": "center",
                },
                xaxis={
                    "title": {"text": "Classification", "font": {"size": 14, "color": "#616161"}},
                    "tickfont": {"size": 12, "color": "#424242"},
                    "gridcolor": "rgba(0,0,0,0.05)",
                    "linecolor": "rgba(0,0,0,0.1)",
                },
                yaxis={
                    "title": {"text": "Probability", "font": {"size": 14, "color": "#616161"}},
                    "tickfont": {"size": 12, "color": "#424242"},
                    "range": [0, 1.1],
                    "gridcolor": "rgba(0,0,0,0.05)",
                    "linecolor": "rgba(0,0,0,0.1)",
                },
                height=450,
                showlegend=False,
                plot_bgcolor="rgba(255,255,255,1)",
                paper_bgcolor="rgba(255,255,255,1)",
                margin={"l": 60, "r": 60, "t": 80, "b": 60},
                font={"family": "Arial, sans-serif"},
            )

        return result_text, fig

    # Example sequences
    examples = [
        ["ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"],
        ["ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG"],
        ["GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCT"],
    ]

    # Custom CSS for modern styling
    custom_css = """
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }

    .dna-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    .result-section {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    .footer-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        border: 1px solid #e9ecef;
    }

    .gradio-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
    }

    .analyze-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 15px 30px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s ease !important;
    }

    .analyze-btn:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }
    """

    with gr.Blocks(
        title="ChimeraLM - Chimeric Read Detector",
        theme=gr.themes.Default(
            primary_hue="blue",
            secondary_hue="gray",
            neutral_hue="slate",
        ),
        css=custom_css,
    ) as interface:
        # Header Section
        with gr.Row():
            gr.HTML("""
                <div class="main-header">
                    <div class="dna-icon">🧬</div>
                    <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">ChimeraLM</h1>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">
                        Advanced Chimeric Read Detection using Deep Learning
                    </p>
                    <p style="margin: 1rem 0 0 0; font-size: 1rem; opacity: 0.8;">
                        Identify chimeric artifacts from whole genome amplification with state-of-the-art accuracy
                    </p>
                </div>
            """)

        # Main Content
        with gr.Row():
            with gr.Column(scale=1):
                # Input Section
                gr.HTML('<div class="input-section">')

                gr.Markdown("""
                ## 📝 Sequence Input

                **How to use:**
                1. Enter your DNA sequence (any length)
                2. Use standard nucleotides: **A**, **C**, **G**, **T**, **N**
                3. Click "Analyze Sequence" to get results
                """)

                sequence_input = gr.Textbox(
                    label="🧬 DNA Sequence",
                    placeholder="Enter your DNA sequence here...\nExample: ACGTACGTACGTACGT...",
                    lines=8,
                    max_lines=15,
                    show_label=True,
                    container=True,
                    scale=2,
                )

                with gr.Row():
                    predict_btn = gr.Button(
                        "🔬 Analyze Sequence", variant="primary", size="lg", elem_classes=["analyze-btn"]
                    )

                gr.Examples(
                    examples=examples, inputs=[sequence_input], label="📚 Example Sequences", elem_id="examples"
                )

                gr.HTML("</div>")

            with gr.Column(scale=1):
                # Results Section
                gr.HTML('<div class="result-section">')

                gr.Markdown("## 📊 Analysis Results")

                result_output = gr.Markdown(
                    value="✨ Enter a sequence and click 'Analyze Sequence' to see detailed results and visualizations.",
                    elem_id="results",
                )

                # Enhanced plot component
                plot_output = gr.Plot(label="📈 Probability Distribution", value=None, elem_id="probability-plot")

                gr.HTML("</div>")

        # Footer Section
        gr.HTML('<div class="footer-section">')

        gr.Markdown(
            """
            ## 🚀 About ChimeraLM

            **Advanced Features:**
            - ⚡ **High Performance**: Optimized for speed and accuracy
            - 🎯 **Binary Classification**: Distinguishes biological vs chimeric sequences
            - 📏 **Long Sequences**: Handles up to 32,768 nucleotides
            - 🤖 **Pre-trained Model**: Ready-to-use with `yangliz5/chimeralm`

            **Technical Specifications:**
            - **Model Type**: Binary Sequence Classifier
            - **Input**: DNA sequences with standard nucleotides
            - **Output**: Classification + confidence scores
            - **Training**: Whole genome amplification artifact detection

            ---

            **📖 Citation:**
            ```
            @software{chimeralm2025,
              title={ChimeraLM: A genomic language model to identify chimera artifacts},
              author={Li, Yangyang, Guo, Qingxiang and Yang, Rendong},
              year={2025},
              url={https://github.com/ylab-hi/ChimeraLM}
            }
            ```

            **🔗 Links:**
            - [GitHub Repository](https://github.com/ylab-hi/ChimeraLM)
            - [Model Hub](https://huggingface.co/yangliz5/chimeralm)
            - [Documentation](https://github.com/ylab-hi/ChimeraLM#readme)
            """
        )

        gr.HTML("</div>")

        # Connect the button click
        predict_btn.click(fn=predict_sequence, inputs=[sequence_input], outputs=[result_output, plot_output])

    return interface


def main():
    """Launch the Gradio interface."""
    logging.basicConfig(level=logging.INFO)
    interface = create_interface()
    interface.launch(share=False)


if __name__ == "__main__":
    main()
