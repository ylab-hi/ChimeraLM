# Web Interface Tutorial

Learn how to use ChimeraLM's interactive web interface for analyzing individual DNA sequences and visualizing predictions in real-time.

!!! info "Learning Objectives"
    By the end of this tutorial, you will be able to:

    - Launch the ChimeraLM web interface
    - Input DNA sequences for analysis
    - Interpret prediction results and confidence scores
    - Understand the visual probability distribution
    - Use example sequences for testing

    **Prerequisites**: ChimeraLM installed, web browser

    **Time**: ~10 minutes

## Overview

The ChimeraLM web interface provides a user-friendly Gradio-based interface for:

- **Sequence Input**: Paste DNA sequences directly into the browser
- **Real-time Prediction**: Get instant classification results
- **Confidence Visualization**: Interactive bar charts showing probabilities
- **Easy to Use**: No command-line experience required
- **Example Sequences**: Pre-loaded examples to get started quickly

!!! note "Use Case"
    The web interface is ideal for **exploring individual sequences**. For analyzing BAM files with thousands of reads, use the [CLI commands](../reference/cli.md) instead.

## Step 1: Launch the Web Interface

Start the web interface with a single command:

```bash
chimeralm web
```

**Expected output:**
```text
Running on local URL:  http://127.0.0.1:7860
```

The interface will automatically open in your default browser. If it doesn't, manually navigate to the URL shown (typically `http://127.0.0.1:7860`).

!!! tip "First Launch"
    The first time you run the web interface, ChimeraLM will download the pretrained model from Hugging Face (`yangliz5/chimeralm`). This may take a few minutes depending on your internet connection.

## Step 2: Understanding the Interface

The web interface has three main sections:

### Header Section

The top banner displays:
- **ChimeraLM logo** (DNA helix icon 🧬)
- **Title and description**
- **Purpose**: "Advanced Chimeric Read Detection using Deep Learning"

### Input Section (Left Panel)

**"📝 Sequence Input"** section includes:

- **Text Area**: Large input box for pasting DNA sequences
- **Valid Characters**: A, C, G, T, N (case-insensitive)
- **Max Length**: Up to 32,768 nucleotides
- **Analyze Button**: Click to run prediction
- **Example Sequences**: Pre-loaded examples to try

### Results Section (Right Panel)

**"📊 Analysis Results"** section shows:

- **Prediction Label**: Biological or Chimeric Artifact
- **Confidence Score**: Probability of the prediction (0-1)
- **Confidence Breakdown**: Probabilities for both classes
- **Probability Chart**: Interactive bar chart visualization

## Step 3: Analyze a DNA Sequence

### Input a Sequence

**Method 1: Type or Paste**

Click in the text area and paste your DNA sequence:

```
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
```

**Method 2: Use Examples**

Click one of the example sequences below the input box:

- Example 1: ACGT repeating pattern
- Example 2: ATCG repeating pattern
- Example 3: GCTA repeating pattern

### Run Prediction

Click the **"🔬 Analyze Sequence"** button to start analysis.

**Processing:**
- Validation of nucleotides
- Tokenization of sequence
- Model inference
- Results display (~1-2 seconds)

!!! note "Valid Characters"
    Only standard DNA nucleotides are accepted:
    - **A** (Adenine)
    - **C** (Cytosine)
    - **G** (Guanine)
    - **T** (Thymine)
    - **N** (Any nucleotide / unknown)

    Both uppercase and lowercase are accepted and will be converted to uppercase.

## Step 4: Interpret Results

### Prediction Output

The results section displays:

**Prediction Example:**
```
**Prediction:** Biological
**Confidence:** 0.892

**Confidence Breakdown:**
- Biological: 0.892
- Chimeric Artifact: 0.108
```

**Understanding the Output:**

- **Prediction**: The model's classification
  - **Biological**: Real genomic sequence (label 0)
  - **Chimeric Artifact**: Artificial sequence from WGA (label 1)

- **Confidence**: Probability score (0.0 to 1.0)
  - **High confidence**: > 0.8 (strong prediction)
  - **Medium confidence**: 0.6 - 0.8 (moderate prediction)
  - **Low confidence**: < 0.6 (uncertain prediction)

- **Confidence Breakdown**: Shows probabilities for both classes
  - Always sums to 1.0 (100%)
  - Helps understand model certainty

### Visual Probability Distribution

The bar chart shows:

- **X-axis**: Two classes (Biological, Chimeric Artifact)
- **Y-axis**: Probability (0.0 to 1.0)
- **Colors**:
  - **Green bar**: Biological prediction (if predicted)
  - **Red bar**: Chimeric Artifact prediction (if predicted)
  - **Gray bar**: Non-predicted class

**Chart Features:**
- **Hover**: Shows exact probability values
- **Interactive**: Pan and zoom
- **Values displayed**: Probabilities shown on bars

### Example Interpretations

**Case 1: High Confidence Biological**
```
Prediction: Biological
Confidence: 0.956
```
→ The sequence is very likely genuine (95.6% probability)

**Case 2: High Confidence Chimeric**
```
Prediction: Chimeric Artifact
Confidence: 0.873
```
→ The sequence is likely a WGA artifact (87.3% probability)

**Case 3: Low Confidence**
```
Prediction: Biological
Confidence: 0.624
```
→ The model is uncertain; consider additional validation

## Step 5: Test with Different Sequences

### Sequence Length Guidelines

**Short Sequences (< 100 bp):**
- May have lower confidence
- Limited context for model

**Medium Sequences (100 - 1000 bp):**
- Good balance of speed and accuracy
- Recommended for testing

**Long Sequences (1000 - 32,768 bp):**
- Highest accuracy
- May take a few seconds longer

### Example Sequences to Try

**Biological-like pattern:**
```
ATGCATGCATGCATGCATGCATGCATGC
```

**Random pattern:**
```
ACGTTAGCCTAAGCCTTAAGCCTAAGCC
```

**Repetitive pattern:**
```
AAAAAACCCCCCGGGGGGTTTTTTAAAA
```

!!! tip "Testing Your Own Sequences"
    Extract sequences from your BAM files using samtools:
    ```bash
    samtools view your_file.bam | head -1 | cut -f10
    ```
    Then paste the sequence into the web interface.

## Advanced Features

### Model Information

The web interface uses:

- **Model**: `yangliz5/chimeralm` (Hugging Face Hub)
- **Architecture**: HyenaDNA-based binary classifier
- **Max Sequence Length**: 32,768 nucleotides
- **Tokenizer**: Character-level (A, C, G, T, N)

### Device Selection

The model automatically uses:

- **GPU** (CUDA) if available → Fastest
- **CPU** if no GPU → Slower but works everywhere

Check the terminal output when launching to see which device is used:
```
Model loaded successfully on cuda
```
or
```
Model loaded successfully on cpu
```

## Troubleshooting

### Invalid Character Error

??? question "Error: Invalid characters in sequence"

    **Problem**: Sequence contains non-ACGTN characters

    **Solution**:
    - Remove spaces, numbers, or special characters
    - Only use: A, C, G, T, N
    - Check for accidental letters (like O vs 0)

    **Example Fix:**
    ```
    ❌ ACG TAG CTG  (spaces not allowed)
    ✅ ACGTAGCTG

    ❌ ACGT123ACGT  (numbers not allowed)
    ✅ ACGTNNACGT   (use N for unknowns)
    ```

### Model Loading Fails

??? question "Error: Failed to load model"

    **Possible causes:**

    1. **No internet connection** (first time only)
       - ChimeraLM needs to download the model
       - Check your internet connection

    2. **Insufficient memory**
       - Model requires ~2GB RAM
       - Close other applications

    3. **GPU out of memory**
       - Model will fall back to CPU automatically
       - Check terminal for device messages

### Empty or No Results

??? question "Results don't appear after clicking Analyze"

    **Solutions:**

    1. **Check sequence length**
       - Minimum: ~10 nucleotides
       - Maximum: 32,768 nucleotides

    2. **Refresh the page**
       - Click browser refresh
       - Re-enter sequence and try again

    3. **Check terminal for errors**
       - Look at the terminal where you launched `chimeralm web`
       - Error messages will appear there

### Port Already in Use

??? question "Error: Address already in use"

    **Problem**: Port 7860 is already in use

    **Solution**:
    ```bash
    # Find what's using the port
    lsof -i :7860

    # Kill the process
    kill <PID>

    # Or just try again (Gradio will auto-select another port)
    chimeralm web
    ```

## Best Practices

### When to Use the Web Interface

✅ **Good use cases:**
- Exploring individual sequences
- Quick testing and validation
- Teaching and demonstrations
- Understanding model behavior
- Checking specific reads of interest

❌ **Not ideal for:**
- Processing thousands of sequences
- Batch analysis of BAM files
- Automated pipelines
- Production workflows

→ **For large-scale analysis**, use the [CLI commands](../reference/cli.md) instead.

### Input Tips

- [ ] **Validate sequence** before submission
- [ ] **Remove whitespace** and special characters
- [ ] **Start with examples** to understand output
- [ ] **Try different lengths** to see accuracy vs sequence length
- [ ] **Compare results** with CLI predictions (should match)

### Interpreting Confidence

**High Confidence (> 0.8):**
- Trust the prediction
- Model is certain about classification

**Medium Confidence (0.6 - 0.8):**
- Prediction is likely correct
- Consider additional validation

**Low Confidence (< 0.6):**
- Model is uncertain
- Manual review recommended
- May need longer sequence or better quality

## Comparison: Web Interface vs CLI

| Feature | Web Interface | CLI (`predict`) |
|---------|--------------|-----------------|
| **Input** | Single DNA sequence | BAM files |
| **Speed** | ~1-2 seconds per sequence | Batch processing |
| **Scale** | 1 sequence at a time | Thousands of reads |
| **Visualization** | Interactive charts | Text file output |
| **Ease of Use** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐ Moderate |
| **Automation** | ❌ Manual only | ✅ Scriptable |
| **Best For** | Exploration, testing | Production, pipelines |

## Technical Details

### Gradio Interface

ChimeraLM uses [Gradio](https://www.gradio.app/) for the web interface:

- **Framework**: Python-based web UI framework
- **Styling**: Custom CSS with gradient backgrounds
- **Charts**: Plotly for interactive visualization
- **Theme**: Modern with purple gradient accents

### Model Architecture

The web interface uses the same model as the CLI:

- **Backbone**: HyenaDNA-small-32k
- **Head**: Binary sequence classifier
- **Training**: Whole genome amplification artifact detection
- **Input**: Tokenized DNA sequences
- **Output**: Binary logits → Softmax probabilities

### Confidence Calculation

```python
# Simplified version of what happens
logits = model(sequence)                    # Raw model output
probabilities = softmax(logits)             # Convert to probabilities
predicted_class = argmax(probabilities)     # Get predicted class (0 or 1)
confidence = probabilities[predicted_class] # Confidence of prediction
```

## Next Steps

- **Batch Processing**: Use [CLI commands](../reference/cli.md#predict-command) for multiple sequences
- **BAM Filtering**: See [Filtering BAM Files](bam-filtering.md) tutorial
- **Integration**: Learn about [Pipeline Integration](pipeline-integration.md)
- **API Access**: Use [Models API](../reference/models.md) for custom workflows

## Summary

You've learned how to:

- ✅ Launch the ChimeraLM web interface
- ✅ Input DNA sequences for analysis
- ✅ Interpret prediction results and confidence scores
- ✅ Understand the probability distribution chart
- ✅ Use example sequences for testing
- ✅ Troubleshoot common issues

!!! success "Ready to Explore!"
    The web interface makes ChimeraLM accessible for quick sequence analysis and exploration. For production workflows with large BAM files, use the CLI commands.

## Additional Resources

- [CLI Commands Reference](../reference/cli.md) - Full command documentation
- [Filtering BAM Files](bam-filtering.md) - Process large datasets
- [Models API](../reference/models.md) - Use ChimeraLM programmatically
- [GitHub Repository](https://github.com/ylab-hi/chimera) - Source code and issues
