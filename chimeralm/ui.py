"""Gradio Web UI for ChimeraLM - Hugging Face Spaces Version."""

import logging

import gradio as gr
import plotly.graph_objects as go
import torch

import chimeralm
from chimeralm.data.tokenizer import load_tokenizer_from_hyena_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers (httpx, huggingface_hub, etc.)
for _lib in ("httpx", "huggingface_hub", "transformers", "urllib3"):
    logging.getLogger(_lib).setLevel(logging.WARNING)


class ChimeraLMPredictor:
    """ChimeraLM predictor for web interface."""

    def __init__(self):
        """Initialize the predictor with model and tokenizer."""
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self._load_model()

    def _load_model(self):
        """Load the ChimeraLM model and tokenizer."""
        try:
            logger.info("Loading ChimeraLM model from Hugging Face Hub...")
            self.model = chimeralm.models.ChimeraLM.from_pretrained("yangliz5/chimeralm")
            self.model.eval()
            self.model.to(self.device)

            logger.info("Loading tokenizer...")
            self.tokenizer = load_tokenizer_from_hyena_model("hyenadna-small-32k-seqlen")
            logger.info(f"✅ Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def predict(self, sequence: str) -> tuple[str, float, dict]:
        """Predict if a DNA sequence is chimeric or biological."""
        if not sequence or not sequence.strip():
            return "Please enter a DNA sequence", 0.0, {}

        # Clean and validate sequence
        sequence = sequence.strip().upper()
        valid_chars = set("ACGTNacgtn")
        if not all(c in valid_chars for c in sequence):
            return "Invalid characters in sequence. Only A, C, G, T, N are allowed.", 0.0, {}

        sequence = sequence.upper()

        try:
            # Tokenize sequence
            tokenized = self.tokenizer(
                sequence,
                truncation=True,
                padding=True,
                max_length=32768,
                return_tensors="pt",
            )

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

            logger.info(f"Prediction: {prediction} (confidence: {confidence:.3f})")
            return prediction, confidence, confidence_breakdown

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return f"Prediction failed: {e}", 0.0, {}


def create_interface():
    """Create the Gradio interface."""
    # Lazy initialization - only load model on first prediction
    predictor_cache: dict[str, ChimeraLMPredictor] = {}

    def _get_predictor() -> ChimeraLMPredictor:
        if "instance" not in predictor_cache:
            predictor_cache["instance"] = ChimeraLMPredictor()
        return predictor_cache["instance"]

    def predict_sequence(sequence):
        prediction, confidence, breakdown = _get_predictor().predict(sequence)

        # Format output with enhanced styling
        if (
            "❌" in prediction
            or "⚠️" in prediction
            or "Please" in prediction
            or "Invalid" in prediction
            or "Prediction failed" in prediction
        ):
            result_text = f"### {prediction}"
        else:
            # Color-coded results with better styling
            color = "#4CAF50" if prediction == "Biological" else "#F44336"
            icon = "✅" if prediction == "Biological" else "⚠️"
            result_text = f"""
### {icon} Prediction Result

<div style="background: {color}; color: white; padding: 1.5rem; border-radius: 15px; text-align: center; margin: 1rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
    <h2 style="margin: 0; font-size: 2rem; font-weight: 700; color: white;">{prediction}</h2>
    <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; color: rgba(255,255,255,0.95);">Confidence: {confidence:.1%}</p>
</div>
"""

        if breakdown:
            result_text += "\n\n### 📊 Detailed Confidence Scores:\n"
            for class_name, prob in breakdown.items():
                emoji = "✅" if class_name == "Biological" else "⚠️"
                prob_value = float(prob)
                result_text += f"- {emoji} **{class_name}**: {prob_value:.1%}\n"

        # Create bar plot with proper contrast
        if breakdown:
            classes = list(breakdown.keys())
            probabilities = [float(prob) for prob in breakdown.values()]

            # Create colors based on prediction with better contrast
            colors = []
            text_colors = []
            for class_name in classes:
                if class_name == prediction:
                    # Vibrant colors for predicted class with white text
                    if prediction == "Biological":
                        colors.append("#4CAF50")  # Green
                    else:
                        colors.append("#F44336")  # Red
                    text_colors.append("white")
                else:
                    # Medium gray for non-predicted class with dark text
                    colors.append("#BDBDBD")
                    text_colors.append("#424242")

            # Create individual bars with appropriate text colors
            bars = []
            for class_name, prob, color, text_color in zip(classes, probabilities, colors, text_colors, strict=True):
                bars.append(
                    go.Bar(
                        x=[class_name],
                        y=[prob],
                        marker_color=color,
                        text=[f"{prob:.1%}"],
                        textposition="auto",
                        textfont={"size": 20, "color": text_color, "family": "Inter, sans-serif", "weight": 600},
                        marker_line={"width": 2, "color": "rgba(255,255,255,0.3)"},
                        width=0.5,
                        opacity=0.95,
                        name=class_name,
                        showlegend=False,
                    )
                )

            fig = go.Figure(data=bars)

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

    # Example sequences - more realistic with varied patterns
    # 1, 1, 0
    examples = [
        [
            "TTGTGTGCCTTCATTAGTTATATACTAGTTCCTGATAAATTCATTTATAGAACAGAAAGACCACAGATTCAATTATATGGAATAGATCTGCTGGTGAATGTAAGAAAGTCTTCTGAACTGCGAAGGGAAAATAAATGATTTAATTCCCACCACCTCTCAACAGCTACCTTCTGTTTTAGAGACACTGGTAAAACTTCTGGGGCTCTTACTTGACATACCTACATCGTATTATAGGCCTATTGGTTTTATCAGAATAATATGCTTTCCTCACATAAGTTATTTCTTTCTGTTACTTGCTTGCAGTACAGATTTAAAGGGGCATTCAGGCAGCCTCCAGATGCCATGATGGATTAACTCTCATGTTACACAGTAATGTAGAAGCTTCTCTTCATTCTCAGACTTTATCTGACAATGAAGAGAAGCTTCTAATTATACTGTGTAAGTTGATCATGTAACACATCTGGAGGCTGCCTGAATGCCCCTTAAATCTGTACTGCAAGCAAGTAACAGAAAGAAATAACTTATGTGAGGAAAGCATATTATTCTGATAAAACCAATACACCCTTATAATACGATGTAGGTATGTCAAGTAAGAGCCCCAGAAGTTTGCAGTATCTAAAACAAAGGTGTTTGTTGAGGTAGTGAGGAAAATAAATCATTTATTTTCCCTTCGCAGTTCAGGCAACACTTTCTTTACATTCACCACCAGATTCCATATAATCTGTGGGAGTCTTTGGCTGTTCTATAAAATGAATTTATCAGTAAATGA"
        ],
        [
            "CAATGGTAAATGAATTCAATAAATATTTGAGGTGATTAAATTTCCTTTCCTAACACATTTTATTTCAAATTCTATTTGAAAGAAAAAATGCTAACAACATAAGAGATCAAATTCAGCTACCTATTTTTTCAACATTCAAATATGCATTAATTGTCTACACTTTGCTAAGCTTGGGCTGATTTCTAGGGCTATAAACATAAATTAAATTTATTCATGGATCTTAAGTGGCTCATGAGCATTAGTACAGCATATTTATAAGCCGAGCATAGTGTCTCATACCTATAATCCCAACACTGGGAGGCTGAGGTGGGAGGATCTCTTGAAGCCGGGAGTTCAAGAACTGCCTGGAAAACATAGCAAGACCCTGTCTCTACCAAAAACAAACAAATAAAACTTAGCCGGGAGTGGCTGCACCTGTAGCTACTCAGGAGTCTGTGATTGGAGGGTAATTTGAACACAGGAGTTTGAGATAGCAGCAAGCTATGATCATGCCACTGTACTCCAGCCTAATTGACAGAACAAGAGCCTGTCTCTAAAATCATTCCATATGTCTATATATAGATATATATATCAAGAAAACTTTACTTTCTAGATTCTAGTTTGTTTTATTGCTCATTCTTTTCTAAATTTATTCATTAGGAGGTATATACAATGTGTTTCAGAGATATAAGAATAGTAAACTTAGAGTGAAAAGGGAAAGATATTTCTTGTTAAAATTCCTAAAATAAAGTATTAAACTTATCTATGAAAAGGCATACATTTCTGTCTGATATTTTATATAAAATAATGGGAACATAATCATATATAATATTTTCTATAAAATGCTTAACAGGTTTTCATAACTTAAATTGTACTTAATATTTTAGGAATTTTAACAATATTCTTCCCTTTTCACTCTAAGTTTACTGTCTTAACCCCCAAAAAACACATTGTCTGTACACCTCCTAATGAATAAATTTAGAAAAAGAAAAAATACAGCAATAAAACAAACTAGTAATACTGGAAGAGTCAAACTTTCTGATATTGTGTACCTCTTCTTATAAAGACATATGGAATGATTTTGAGGACAGGTATTGTTCTGATTAGGCTGGAGTACAGTGGCATGATCATAGCTTACTGCTATCTCGAACTCCTGTGTTAAATTCTCTCCAATCACAGACTCCTGAGTAGCTACAGGTGAGCCACTGCCCGGCTAAGTTTTATTTGTTTTGTTTTTGGTAGAGACAGGGTCTTGCTATGTTTGCCAGGCTGGTCTTGAACTCCCGGCTTCAAGAGATCCTCCCACCTCAGCCTCCCAGTGTTGGGATTATAGGTATGAGACACTATGCTCAGCTAACAAATATATAATGCTCATGAGCCACTAATCAAGTCAAGAATTTAAATTTATGTTTATAGCCCCATCAGCCCCAAGCTTAGCAAAGTGTAGACAATTAATGTAACATTTGAATGCTGAAAAAATAGGTATAGAAATTTGATCTTACCCTATATTGTAGCATTTTTTTCTTTCAAATAAATTTGAAATAAAATGTGTAGGGAAAAGGAAATTAAATCACCTCAACATTTTATAAAAATCATTTACCATTGGCTAT"
        ],
        [
            "ATGTTGTGTACCTGGTTCGGTTCGTCTATGGTATGCACCTTGGCTATCATCACCCGATGAGGCAACCAGCCGGGAGACACCTAAACCCATCATCTCCTGTACCACCCTAGTAGGCTCCCTTCCCCTACTCATCGCACTAATTTACACTCACAACACCCTAGGCTCACTAAACATTCTACTACTCACTCTCACTGCCCAACTAAACTCCTGGCCATCCCCTTATGAGCGGGCGCAGTGATTATAGGCTTTCGCTCTAAGATTAAAAATGCCCTAGCCCACTTCTTACCACAAGGCACACCTACACCCCTTATCCCCATACTGGCTGTTGTGAAAACCATAGCCTACTATCGTTCAACAATAGCCCTGGCCGTACGCCTAACCGCTAACATTACTGCAGGCCACCTACTCATGCACCTAATTGGAAGCGCCACCCTAGCAATATCAACCATTAACCTTACCTACACTTATAGTCTTTCACAATTCTAATTCTACTGACTATCCTAGAAATCGCTGTCGCCTTAATCCAAGCCTACGTTTTCACACTTCTAGTAAGCCTCTACCTGCACGACAACACATAATGACCCACCAATCACATGCCTATCATGGCTAAACCCAGCCCATGACCCCTAACAGGGGCCCTCTCAGCCCTCCTAATGACCTCCGGCCTAGCCATGTGATTTCACTTCCACTCCATAACGCTCCTCATACTAGGCCTACTAACCAACACACTAACCATATACCAATAATGGCAATGTAACGCAAAGCACATACCAAGGCCACCACACACCACCTCTATTAAAAAGGCC"
        ],
    ]

    # Custom CSS for modern, visually appealing styling
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Global text color improvements */
    body {
        background: linear-gradient(135deg, #f5f7fa 0%, #e3e7ed 100%);
        min-height: 100vh;
    }

    /* Ensure all headings have good contrast */
    h1, h2, h3, h4, h5, h6 {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    /* Ensure all paragraphs and text have good contrast */
    p, li, span, div {
        color: #37474F !important;
    }

    /* Universal text color fix for all content */
    strong, b {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    /* Ensure all text in Gradio blocks has proper contrast */
    .gradio-block p, .gradio-block li, .gradio-block span,
    .gradio-block div, .gradio-block strong, .gradio-block b {
        color: #37474F !important;
    }

    .gradio-block h1, .gradio-block h2, .gradio-block h3,
    .gradio-block h4, .gradio-block h5, .gradio-block h6,
    .gradio-block strong, .gradio-block b {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    /* Label styling */
    label {
        color: #2C3E50 !important;
        font-weight: 600 !important;
    }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3rem 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        bottom: -50%;
        left: -50%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }

    @keyframes shine {
        0% { transform: translateX(-100%) rotate(45deg); }
        100% { transform: translateX(100%) rotate(45deg); }
    }

    .dna-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: pulse 2s ease-in-out infinite;
        display: inline-block;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.2));
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    .input-column {
        background: white;
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        margin: 0.5rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .input-column:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    /* Ensure input column text has good contrast */
    .input-column h1, .input-column h2, .input-column h3,
    .input-column h4, .input-column h5, .input-column h6 {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    .input-column p, .input-column li, .input-column span,
    .input-column div, .input-column strong, .input-column b,
    .input-column code, .input-column pre {
        color: #37474F !important;
    }

    /* Ensure markdown content in input column has proper colors */
    .input-column .markdown, .input-column .markdown *,
    .input-column [class*="markdown"], .input-column [class*="markdown"] * {
        color: #37474F !important;
    }

    .input-column .markdown h1, .input-column .markdown h2, .input-column .markdown h3,
    .input-column [class*="markdown"] h1, .input-column [class*="markdown"] h2, .input-column [class*="markdown"] h3 {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    .input-column .markdown strong, .input-column .markdown b,
    .input-column [class*="markdown"] strong, .input-column [class*="markdown"] b {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    .result-column {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 20px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.1);
        margin: 0.5rem;
        border: 1px solid rgba(102, 126, 234, 0.1);
        min-height: 500px;
    }

    /* Ensure text readability in result column */
    .result-column h1, .result-column h2, .result-column h3,
    .result-column h4, .result-column h5, .result-column h6 {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    .result-column p, .result-column li, .result-column span,
    .result-column div, .result-column markdown {
        color: #37474F !important;
    }

    /* Markdown content styling - comprehensive */
    .markdown, .markdown *, [class*="markdown"], [class*="prose"] {
        color: #37474F !important;
    }

    .markdown h1, .markdown h2, .markdown h3,
    .markdown h4, .markdown h5, .markdown h6,
    [class*="markdown"] h1, [class*="markdown"] h2, [class*="markdown"] h3,
    [class*="markdown"] h4, [class*="markdown"] h5, [class*="markdown"] h6 {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    .markdown p, .markdown li, .markdown span,
    .markdown div, .markdown code, .markdown pre,
    .markdown strong, .markdown b,
    [class*="markdown"] p, [class*="markdown"] li, [class*="markdown"] span,
    [class*="markdown"] div, [class*="markdown"] strong, [class*="markdown"] b {
        color: #37474F !important;
    }

    .markdown code, [class*="markdown"] code {
        background: #f5f7fa !important;
        color: #2C3E50 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }

    /* Target all Gradio markdown blocks */
    .gradio-markdown, .gradio-markdown *,
    div[class*="markdown"], div[class*="prose"] {
        color: #37474F !important;
    }

    div[class*="markdown"] h1, div[class*="markdown"] h2, div[class*="markdown"] h3,
    div[class*="markdown"] h4, div[class*="markdown"] h5, div[class*="markdown"] h6 {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    div[class*="markdown"] strong, div[class*="markdown"] b,
    div[class*="markdown"] p, div[class*="markdown"] li,
    div[class*="markdown"] span, div[class*="markdown"] div {
        color: #37474F !important;
    }

    .footer-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-top: 2rem;
        border: 2px solid #dee2e6;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
    }

    /* Ensure footer text has good contrast */
    .footer-section h1, .footer-section h2, .footer-section h3,
    .footer-section h4, .footer-section h5, .footer-section h6 {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    .footer-section p, .footer-section li, .footer-section span,
    .footer-section div, .footer-section a, .footer-section code,
    .footer-section strong, .footer-section b {
        color: #37474F !important;
    }

    /* Ensure markdown content in footer has proper colors */
    .footer-section .markdown, .footer-section .markdown *,
    .footer-section [class*="markdown"], .footer-section [class*="markdown"] * {
        color: #37474F !important;
    }

    .footer-section .markdown h1, .footer-section .markdown h2, .footer-section .markdown h3,
    .footer-section [class*="markdown"] h1, .footer-section [class*="markdown"] h2, .footer-section [class*="markdown"] h3 {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    .footer-section .markdown strong, .footer-section .markdown b,
    .footer-section [class*="markdown"] strong, .footer-section [class*="markdown"] b {
        color: #2C3E50 !important;
        font-weight: 700 !important;
    }

    .footer-section a {
        color: #667eea !important;
        text-decoration: none !important;
        font-weight: 600 !important;
    }

    .footer-section a:hover {
        color: #764ba2 !important;
        text-decoration: underline !important;
    }

    .footer-section code {
        background: #f5f7fa !important;
        color: #2C3E50 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
        border: 1px solid #e0e0e0 !important;
    }

    .gradio-container {
        max-width: 1400px !important;
        margin: 0 auto !important;
        padding: 2rem 1rem !important;
    }

    .analyze-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 30px !important;
        padding: 18px 40px !important;
        font-size: 17px !important;
        font-weight: 600 !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .analyze-btn:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.6) !important;
    }

    .analyze-btn:active {
        transform: translateY(-1px) scale(0.98) !important;
    }

    /* Enhanced textbox styling */
    textarea {
        border: 2px solid #e9ecef !important;
        border-radius: 12px !important;
        font-family: 'Courier New', monospace !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        transition: border-color 0.3s ease !important;
        background-color: #ffffff !important;
        color: #2C3E50 !important;
        padding: 14px !important;
    }

    textarea::placeholder {
        color: #90A4AE !important;
        opacity: 0.8 !important;
    }

    textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        background-color: #ffffff !important;
        outline: none !important;
    }

    /* Info cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        transition: transform 0.2s ease;
    }

    .info-card:hover {
        transform: translateX(5px);
    }

    /* Examples styling */
    #examples {
        border-radius: 15px;
        overflow: hidden;
        margin-top: 1.5rem;
    }

    /* Enhanced examples button styling */
    .example-btn {
        background: #f8f9fa !important;
        border: 2px solid #e0e0e0 !important;
        color: #2C3E50 !important;
        border-radius: 8px !important;
        padding: 12px 20px !important;
        transition: all 0.3s ease !important;
    }

    .example-btn:hover {
        background: #667eea !important;
        border-color: #667eea !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }

    /* Better spacing and visual hierarchy */
    .gradio-block {
        margin-bottom: 1.5rem;
    }

    /* Improved scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
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
                    <h1 style="margin: 0; font-size: 3rem; font-weight: 700; position: relative; z-index: 1;">ChimeraLM</h1>
                    <p style="margin: 0.5rem 0 0 0; font-size: 1.3rem; opacity: 0.95; font-weight: 500; position: relative; z-index: 1;">
                        Advanced Chimeric Read Detection using Deep Learning
                    </p>
                    <p style="margin: 1rem 0 0 0; font-size: 1.05rem; opacity: 0.85; position: relative; z-index: 1;">
                        Identify chimeric artifacts from whole genome amplification with state-of-the-art accuracy
                    </p>
                    <div style="margin-top: 1.5rem; position: relative; z-index: 1;">
                        <span style="display: inline-block; background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.25rem; font-size: 0.9rem;">
                            ⚡ High Performance
                        </span>
                        <span style="display: inline-block; background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.25rem; font-size: 0.9rem;">
                            🎯 98% Accuracy
                        </span>
                        <span style="display: inline-block; background: rgba(255,255,255,0.2); padding: 0.5rem 1rem; border-radius: 20px; margin: 0.25rem; font-size: 0.9rem;">
                            🚀 Pre-trained
                        </span>
                    </div>
                </div>
            """)

        # Main Content
        with gr.Row():
            with gr.Column(scale=1, elem_classes="input-column"):
                # Input Section
                gr.Markdown("""
                ## 📝 DNA Sequence Input

                **Quick Start Guide:**
                1. 🧬 Enter your DNA sequence (supports up to 32,768 bp)
                2. ✅ Use standard nucleotides: **A**, **C**, **G**, **T**, **N**
                3. 🔬 Click "Analyze Sequence" for instant results
                4. 📊 View confidence scores and visualization below

                **What is Chimeric DNA?**
                Chimeric reads are artificial DNA sequences created during whole genome amplification (WGA),
                where fragments from different genomic locations are incorrectly joined together.
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

            with gr.Column(scale=1, elem_classes="result-column"):
                # Results Section

                gr.Markdown("## 📊 Analysis Results")

                result_output = gr.Markdown(
                    value="✨ Enter a sequence and click 'Analyze Sequence' to see detailed results and visualizations.",
                    elem_id="results",
                )

                # Enhanced plot component
                plot_output = gr.Plot(label="📈 Probability Distribution", value=None, elem_id="probability-plot")

        # Footer Section
        with gr.Row():
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
                """,
                elem_classes="footer-section",
            )

        # Connect the button click
        predict_btn.click(fn=predict_sequence, inputs=[sequence_input], outputs=[result_output, plot_output])

    return interface


# Create demo instance for Gradio auto-reload compatibility
demo = create_interface()

# Enable queueing to handle requests properly and avoid Content-Length issues
demo.queue(max_size=20)

if __name__ == "__main__":
    logger.info("🚀 Starting ChimeraLM Web Interface...")

    # Launch with proper server configuration
    demo.launch(share=False, show_error=True)
