## Installation

The step-by-step installation instructions are shown below.

```bash
git clone https://github.com/webber2933/ST-CLIP.git
cd ST-CLIP/ZS_JHMDB/ST-CLIP

conda create -n STCLIP python=3.7
conda activate STCLIP
pip install -r requirements.txt
pip install -e .
```

If you encounter the following problem in this step `pip install -r requirements.txt`:

```bash!
Cargo, the Rust package manager, is not installed or is not on PATH. This package requires Rust and Cargo to compile extensions. Install it through the system's package manager or via https://rustup.rs/
```
Please try `curl https://sh.rustup.rs -ssf | sh`

