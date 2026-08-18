# Dataset

The published study uses the **Saarbrücken Voice Database (SVD)**.

The research paper describes an experimental subset containing 2,041 /a/ vowel recordings, with 687 healthy and 1,354 pathological samples.

The dataset is **not included in this repository**.

## Local setup

1. Obtain SVD through its authorized access route.
2. Keep the recordings outside GitHub if redistribution is not permitted.
3. Point the notebook or training script to your local dataset directory.
4. Verify that the local labels and file structure match the loader you intend to use.

The repository's loader also supports generic CSV, Excel, JSON, and common audio inputs. That generic loader is not a substitute for the exact SVD multimodal data preparation described in the paper.
