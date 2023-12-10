# ML-Stock-Market-Prediction

We recommend using Google Colab to run the code since all requirements come preinstalled. Simply open a new Colab enviornment and copy&paste the commands below. No GPU is required so you can run everything Colab's CPU. The repository includes the results of our last run (logfile and plots). Running the snippet below will produce new results (may be slightly different due to random initialization of weights).

```bash
!git clone https://github.com/SBodapati11/ML-Test_Repo.git
print("\n\n\n")
!python ML-Test_Repo/src/main.py

from IPython.display import Image, display
display(Image('ML-Test_Repo/plots/original_plot.png'))
print("\n\n\n")
display(Image('ML-Test_Repo/plots/sequenced_plot.png'))
print("\n\n\n")
display(Image('ML-Test_Repo/plots/training_errors.png'))
print("\n\n\n")
!cat ML-Test_Repo/gru_logs.txt
```