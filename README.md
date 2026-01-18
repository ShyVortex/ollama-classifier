# neural-clap
### Neural Crowd Listener for App Release Planning

> **Evolution of the CLAP project:** reimagining mobile app review classification using Large Language Models (LLMs).

---

## About
**neural-clap** is an advanced CLI classification suite designed to analyze user feedback from mobile app stores.
Building upon the methodology introduced in [*"Listening to the Crowd for the Release Planning of Mobile Apps"*](https://ieeexplore.ieee.org/document/8057860) (*Scalabrino et al., 2017*),
this repository replaces the original Random Forest and manual NLP pipelines with state-of-the-art LLMs.

While the original **CLAP** tool relied on statistical methods, stemming, and custom negation handling,
**neural-clap** leverages the semantic reasoning of Generative AI to categorize reviews with greater nuance, specifically
targeting complex categories like usability and security where statistical models traditionally struggled.

## Key Features
* **Hybrid Model Architecture:** run classifications using local privacy-focused models (via [**Ollama**](https://github.com/ollama/ollama))
  or separately with powerful cloud-based models (e.g. *[ChatGPT](https://chatgpt.com/), [Gemini](https://gemini.google.com/), [Mistral](https://chat.mistral.ai/chat), [Qwen](https://qwen.ai/)*).
* **7-Category Standard:** automatically sorts reviews into the rigorous taxonomy defined in the original research paper:
    * **BUG**: crashes, broken functionality, errors.
    * **FEATURE**: requests for new capabilities.
    * **PERFORMANCE**: lag, slow loading, freezes.
    * **SECURITY**: privacy concerns, hacks, permissions.
    * **ENERGY**: battery drain, overheating.
    * **USABILITY**: UI difficulties, accessibility, confusing design.
    * **OTHER**: non-informative reviews, praise, or noise.
* **Classification Arena:** includes scripts to **compare and benchmark** results between CLAP and different models
  to evaluate accuracy against the original baseline.
* **Sample Generation**: create a sample from an input dataset according to the desired Confidence Level and Confidence
  Interval.
* **Sample Analyzer**: analyze and compare a manually annotated sample with an automatically labeled one, using metrics
  like *Accuracy*, *Precision*, *Recall* and *F1-score*.

## CLAP vs. Neural-CLAP
| Feature               | CLAP (2017)                           | Neural-CLAP (2026)                    |
|:----------------------|:--------------------------------------|:--------------------------------------|
| **Core Engine**       | Random Forest (Statistical)           | Large Language Models (Generative)    |
| **Preprocessing**     | Heavy (Stop-words, Stemming, N-grams) | Minimal / None (Raw Text)             |
| **Negation Handling** | Custom State Machine / Parser         | Native Semantic Understanding         |
| **Multilingual**      | Failed (~50% accuracy loss)           | Native Multilingual Support (via LLM) |

You can check the results of this research in "*data/benchmarks*".

## Installation
First, head over to [Ollama's website](https://ollama.com/download/linux) and follow the instructions to install
it on your running system.
Ollama is a software that makes it easier to run local LLMs on any machine.
Download and install a model you want from their [library](https://ollama.com/library).

In a terminal window, type the following to set up the project:
```bash
git clone https://github.com/ShyVortex/neural-clap.git
cd neural-clap
pip install -r requirements.txt
```

## Usage
Run the classifier leveraging local LLMs:
```bash
# Basic usage (use CLAP's dataset, defaults to no reasoning)
python src/classifier.py --model [MODEL_NAME] --prompt [PROMPT_PATH]

# Complete usage (use your own dataset, decide on reasoning)
python src/classifier.py --data [DATASET_PATH] --model [MODEL_NAME]
                         --prompt [PROMPT_PATH] --reasoning [Y/N]
```
You can add your own prompts in the 'prompts' folder.

Run the match detector script:
```bash
python src/match_detector.py --clap [CLAP_PATH] --model-family [FAMILY_NAME]
                             --prediction-source [CHOICE] (dataset / sample)
                             --model [JSON_FILENAME]
```

Run the sample analyzer script:
```bash
# Sample generation usage
python src/sample_analyzer.py --dataset [DATASET_PATH] --sample no
                              --level [CONFIDENCE_LEVEL] (90 || 95 || 99)
                              --interval [CONFIDENCE_INTERVAL] (max value: 100.0)

# Analysis of metrics usage
python src/sample_analyzer.py --dataset [MANUAL_SAMPLE_PATH] --sample yes
                              --compare [AUTOMATIC_SAMPLE_PATH]
```

## License
- This project is distributed under the [GNU General Public License v3.0](https://github.com/ShyVortex/neural-clap/blob/main/LICENSE).
- Copyright of [@ShyVortex](https://github.com/ShyVortex) and [@garganos1](https://github.com/garganos1), 2026.
