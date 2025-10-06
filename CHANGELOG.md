## [unreleased]

### ⚙️ Miscellaneous Tasks

- Update README and GitHub Actions workflow for improved clarity and consistency in publishing process
## [1.0.0] - 2025-10-06

### 🚀 Features

- Update project name and dependencies versions
- Add deepbiop dependency and test_deepbiop function
- Add new Rust file with hello world program
- Add new functionality for annotating chimeric events
- Update dependencies versions for walkdir, rayon, pyo3, noodles, bstr, lazy_static, tempfile, parquet, arrow, and flate2
- Add new file summary.rs and update code in annotate.rs
- Add overlap checking function
- Add functionality to write results to file
- Add chimeric events query functionality
- Add option to output chimeric events
- Update noodles version to 0.84.0
- Add extraction of chimeric reads from BAM file
- Add debug log for each line in worker function
- Add debug print statement in add function
- Add script to annotate structural variant events
- Add Transformer model with encoder-decoder architecture
- Add fire library as a dependency and use it in run script
- Introduce KmerTokenizer and refactor Tokenizer class; update run script to use CharacterTokenizer
- Enhance target name parsing with validation and add tests for data module
- Add module docstrings and improve comments for clarity
- Add Mamba and LocalAttention models for DNA sequence classification; update tokenizer to use 'labels' key
- Implement DNAConvNet model for DNA sequence classification with convolutional layers and one-hot encoding
- Add DNAConvNet and Mamba models for DNA sequence classification; restructure components
- Update models to use embedding layers instead of one-hot encoding; add vocab_size and padding_idx parameters
- Update training and model configuration settings
- Implement MambaSequenceClassification model for sequence classification tasks
- Refactor tokenizer initialization and update Mamba model configurations
- Update Mamba model configuration and refactor parameters for improved flexibility
- Simplify MambaSequenceClassification docstring and update model configuration paths
- Enhance tokenizer functions to support optional quality input and improve data alignment
- Update model configurations and dependencies for improved compatibility and performance
- Add tests for data module with left and right padding configurations
- Add MambaSequenceClassificationSP module
- Add configuration files for MambaSequenceClassificationSP model and experiment setup
- Add SequenceAnalyzer class for sequence analysis and visualization
- Implement CLI for DeepChopper with prediction capabilities and custom output handling
- Enable callback instantiation in evaluation script and set Tensor Core precision for NVIDIA GPUs
- Add candle-core dependency and implement prediction loading functions in Rust
- Add function to write predicts to file
- Add script for predicting with structural variants
- Compare chimeric read
- Implement functions for handling primary chimeric alignments and update project configuration
- Enhance alignment segment handling and add chimeric read comparison functionality
- Add functionality to compare chimeric reads with supporting chimeric reads and output results
- Refactor alignment segment sorting and enhance comparison functionality with threshold parameter
- Add strand comparison to alignment segment comparison function
- Update overlap threshold default value and enhance sorting of chimeric event intervals
- Add data selection script for training, validation, and testing with support-based grouping
- Enhance data selection output filenames to include data counts
- Add print statement to display grouped data by support in selection function
- Add validation checks for sufficient positive and negative data in selection function
- Refactor data selection function to improve readability and add output directory support
- Add print statement to display count of reads grouped by support in selection function
- Replace aligntools with pyfastx and add extract script for chimeric reads
- Enhance extract script with logging, input validation, and output directory support
- Optimize file handling in extract script by separating FASTQ file context management
- Add hyena
- Add cnn and transformer
- Revise transformer
- Train models using new dataset
- Train models using new dataset
- Update dependencies and enhance data processing with new OnlyFqDataModule for FastQ files
- Enhance BAM filtering function with sorting and indexing options
- Add version option and command ordering to Chimera CLI; suppress ImportError for mamba_ssm
- Implement BamDataModule for BAM file processing and update main script to utilize it
- Add ChimeraLM model class and update prediction script for improved model loading and output handling
- Update data module to include BamDataModule in the initialization and export list
- Add hf-xet dependency and refactor BAM file handling in main script
- Enhance BinarySequenceClassifier with attention weight saving and optimized pooling methods
- Add verbose option to BAM file filtering function for enhanced logging
- Add summary logging for prediction counts in BAM file filtering function
- Add Tensor Core precision setting for NVIDIA GPUs in prediction workflow
- Add Gradio web interface for ChimeraLM model and include Plotly for visualization
- Add checkpoint loading option to predict function and implement new model initialization method in ChimeraLM
- Implement comprehensive statistical analysis framework for attention patterns, including position significance testing, multiple testing correction, and visualization methods
- Enhance logging for model loading in predict function to improve user feedback
- Update help text for ChimeraLM and enhance predict function options with checkpoint path and improved verbosity
- Add joblib for parallel processing in prediction loading and filtering functions, enhancing performance and efficiency
- Enhance BAM filtering function by adding num_workers parameter for improved parallel processing

### 🐛 Bug Fixes

- Update branch to 'dev' in deepbiop dependency
- Update pyo3 version to 0.21.2
- Update pre-commit hooks versions to latest versions
- Correct spelling of "Artificial" in docstring; update model initialization and configuration files for consistency
- Update batch size to 10
- Update batch size to 8
- Update batch size to 48 in mambasp.yaml
- Update model and data batch sizes to 128 and 64
- Increase patience for early stopping and model training in YAML configurations
- Adjust patience parameter to 4 and update weight decay range in YAML configurations
- Clean up evaluation script and update model configuration in YAML files
- Update batch size in YAML configuration for training data
- Refactor input_quals handling and clean up configuration files
- Improve resume_read_name function and update label attribute in Predict class
- Clean up commented paths in model notebook and remove unnecessary whitespace in Rust code
- Correct FASTQ entry format by adding '@' prefix to read name in write_fastq_entry function
- Update data paths in mambasp configuration for training and validation datasets
- Add error handling and resource management in BAM file filtering function
- Add warning for empty predictions and improve progress bar integration in predict function
- Ensure output directory is created if it does not exist in PredictionWriter
- Update command in GitHub Actions workflow from deepchopper to chimeralm for consistency

### 💼 Other

- Add model checkpointing and early stopping to predict function, and update output path handling for predictions
- Reset execution counts and refresh notebook metadata in attention.ipynb for improved clarity

### 🚜 Refactor

- Improve sorting of intervals in chimeric events
- Remove unnecessary imports and code lines
- Update references from deepchopper to chimera
- Update function annotations and imports in train.py
- Improve error handling and code structure
- Remove unused code in annotatesv.rs
- Update default values for model_max_length
- Simplify tokenization logic and improve handling of quality scores
- Remove unused imports and fix code formatting
- Clean up imports and improve model configuration in hyena.yaml
- Remove obsolete run.py script
- Update configuration files for hyena experiment and model settings
- Change loss function to CrossEntropyLoss in hyena.yaml
- Update tokenizer target path in hyena.yaml
- Include hyena component in model exports
- Correct class name from HeynaDna to HyenaDna in hyena.py
- Rename Classifier to BinarySequenceClassifier and enhance architecture for sequence classification
- Log logits and labels shapes in ClassificationLit; update output layer for binary classification in BinarySequenceClassifier
- Replace logging of logits and labels shapes with print statements in ClassificationLit
- Remove print statements for logits and labels shapes in ClassificationLit
- Use Path.open for file handling in load_predicts function
- Remove Predict module and related functions from the Rust implementation, streamlining the codebase
- Remove QUAL_FEATURE from BamDataModule data processing to streamline dataset handling
- Update BamDataModule to remove ID_FEATURE instead of QUAL_FEATURE for improved dataset processing
- Enhance BinarySequenceClassifier initialization with optimized activation handling and attention pooling
- Update BinarySequenceClassifier to use instance variable for attention weight saving
- Update BamDataModule to remove ID_FEATURE from dataset processing for improved efficiency
- Update tokenizer initialization in main script to load from Hyena model for improved compatibility
- Update prediction function to allow non-deterministic behavior and improve BAM file filtering options
- Simplify BAM file filtering function and add command-line interface for filtering by predictions
- Rename package from 'chimera' to 'chimeralm' and restructure module organization for improved clarity and maintainability
- Rename 'random_seed' option to 'random' in predict function for clarity
- Update import statements in eval and train scripts to use 'chimeralm' and improve code organization
- Enhance logging and type annotations in prediction and BAM filtering functions
- Update predict function to include shorthand option for 'random' parameter and enhance from_pretrained method with attention saving option
- Integrate RankedLogger for enhanced logging and improve logging consistency in prediction and BAM filtering functions
- Update Gradio UI for ChimeraLM with enhanced styling, improved error handling, and updated plot configurations
- Extract device determination logic into a separate function for cleaner code in prediction workflow
- Simplify prediction loading functions by removing parallel processing and updating function signatures for clarity
- Update model and data module targets to use chimeralm namespace for consistency across configuration files
- Improve error handling in load_predicts function and update predict function docstring for clarity
- Simplify output directory handling in PredictionWriter by removing dataloader index from path
- Streamline folder creation logic in PredictionWriter by removing caching and ensuring direct output directory usage
- Remove limit_predict_batches option from predict function and update logging for output path filtering
- Update prediction output path handling and adjust num_workers for BamDataModule to improve prediction performance
- Streamline logging for prediction output path and adjust num_proc handling in BamDataModule for improved efficiency
- Add collect_txt_from_file function for improved handling of prediction files and update filter command to include summary option
- Update callback handling in predict function to conditionally include training callbacks based on checkpoint path
- Enhance filter_bam_by_prediction function to optionally output predictions to a text file and update filter command to include new output option
- Add logging for prediction output file path in filter_bam_by_prediction function
- Reorganize model loading and output path handling in predict function for improved clarity and efficiency
- Update execution counts, fix weight slicing, and add statistical results saving function in attention.ipynb

### 📚 Documentation

- Add documentation on chimeric events extraction
- Add annotations to bin files
- Remove outdated test workflow
- Update query.rs documentation
- Add docstrings for Tokenizer methods
- Add missing documentation for new features and updates

### 🎨 Styling

- Fix typo in variable names in annotate.rs
- Update .gitignore for notebook file
- Fix spelling errors in comments
- Update pyproject.toml formatting
- Update pyproject.toml formatting
- Remove unnecessary metadata in DEA notebook
- Simplify input_quals handling in forward method
- Update RAM and GPU settings in run_train.sh
- Remove unnecessary comments and environment variable settings
- Update default values for Mamba component

### ⚙️ Miscellaneous Tasks

- Add tmp/ to .gitignore
- Update excluded files in pyproject.toml
- Update dependencies and fix function calls in addtarget.rs
- Update dependencies and refactor FqDataModule; remove unused MNISTDataModule
- Update pre-commit configuration and refactor imports; clean up notebook and script formatting
- Move deepbiop dependency to the correct position in pyproject.toml
- Update pre-commit configuration and add torchvision dependency
- Update dependencies in pyproject.toml and add cache keys for build isolation
- Update project configuration and prepare for future enhancements
- Update dependencies and improve logging syntax across multiple files
- Update pre-commit hooks to latest versions for improved linting and code quality
- Upgrade pre-commit denpdencies
- Update pre-commit configuration to ruff v0.13.1, add missing imports in eval and train scripts, and enhance logging condition in RankedLogger
- Update citation year in README from 2024 to 2025
- Add statsmodels dependency for enhanced statistical analysis in development environment
- Add BAM data module configuration for data loading and processing
- Update .gitignore, bump ruff version in pre-commit config, add GitHub labels, and remove outdated workflows
- Add GitHub Actions workflow for Python library release, including multi-platform support and testing
- Migrate build system from Maturin to Hatchling and update project metadata
- Simplify GitHub Actions workflow for package testing and publishing, transitioning from Maturin to Hatchling with streamlined steps
- Enhance GitHub Actions workflow for package release by adding build and publish steps for TestPyPI and PyPI, with improved branch and tag handling
- Update .gitignore to remove uv.lock and add new uv.lock file for dependency management
- Remove Python 3.12 from GitHub Actions workflow matrix for package release
- Restrict Python version to <3.12 in pyproject.toml for compatibility
- Remove Maturin configuration from pyproject.toml as part of the transition to Hatchling
- Update GitHub Actions workflow to streamline TestPyPI publishing by consolidating token usage in the publish command
- Update GitHub Actions workflow to use latest version of uv for improved publishing functionality
- Update GitHub Actions workflow to use uv@v6 for enhanced publishing capabilities
- Bump version to 1.0.0 and update Python version requirement to >=3.10,<3.12 in pyproject.toml
