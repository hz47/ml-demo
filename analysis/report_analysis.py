import matplotlib.pyplot as plt
import numpy as np

from analysis.distribution import run_distribution
from analysis.sms_length import run_sms_length
from analysis.sms_word_count import run_sms_word_count
from analysis.sms_sentence_count import run_sms_sentence_count
from analysis.word_analysis_stop import run_word_stop_analysis
from analysis.correlation import compute_correlation_matrix, compute_correlation_by_label


def create_bar(ax, x_labels, values, title, ylabel, color):
    x = np.arange(len(x_labels))
    ax.bar(x, values, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")


def create_hbar(ax, words, title, color):
    y_pos = np.arange(len(words))
    ax.barh(y_pos, range(len(words), 0, -1), color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("Rank")
    ax.set_title(title, fontweight="bold")


def create_heatmap(ax, corr_matrix, title, cmap="coolwarm"):
    short_names = {"text_length": "Length", "word_count": "Words", "sentence_count": "Sentences"}
    columns = [short_names.get(c, c) for c in corr_matrix.columns]
    
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title(title, fontweight="bold")

    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10, color=color, fontweight="bold")
    return im


def create_report():
    dist = run_distribution()
    length_stats = run_sms_length()
    word_counts = run_sms_word_count()
    sentence_counts = run_sms_sentence_count()
    word_results = run_word_stop_analysis(top_n=10)
    corr_overall = compute_correlation_matrix()
    corr_by_label = compute_correlation_by_label()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("SMS Spam Analysis Report", fontsize=16, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(3, 3, 1)
    ax1.pie(dist.values, labels=dist.index, autopct="%1.1f%%",
            colors=["#66b3ff", "#ff6666"], explode=(0.02, 0.02), startangle=90)
    ax1.set_title("Label Distribution", fontweight="bold")

    ax2 = fig.add_subplot(3, 3, 2)
    x = np.arange(len(length_stats.index))
    ax2.bar(x - 0.175, length_stats["mean"], 0.35, label="Mean", color="#3498db")
    ax2.bar(x + 0.175, length_stats["median"], 0.35, label="Median", color="#2ecc71")
    ax2.set_xticks(x)
    ax2.set_xticklabels(length_stats.index)
    ax2.set_ylabel("Characters")
    ax2.set_title("Text Length by Label", fontweight="bold")
    ax2.legend()

    create_bar(fig.add_subplot(3, 3, 3), word_counts.index, word_counts.values,
               "Word Count by Label (Median)", "Word Count", "#9b59b6")

    create_bar(fig.add_subplot(3, 3, 4), sentence_counts.index, sentence_counts.values,
               "Sentence Count by Label (Median)", "Sentence Count", "#f39c12")

    create_hbar(fig.add_subplot(3, 3, 5), word_results["spam"], "Top 10 Words (Spam)", "#ff6666")
    create_hbar(fig.add_subplot(3, 3, 6), word_results["ham"], "Top 10 Words (Ham)", "#66b3ff")

    create_heatmap(fig.add_subplot(3, 3, 7), corr_overall, "Correlation (Overall)", "coolwarm")
    im8 = create_heatmap(fig.add_subplot(3, 3, 8), corr_by_label.get("ham", corr_overall), "Correlation (Ham)", "coolwarm")
    create_heatmap(fig.add_subplot(3, 3, 9), corr_by_label.get("spam", corr_overall), "Correlation (Spam)", "coolwarm")

    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.3)
    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.20])
    cbar = fig.colorbar(im8, cax=cbar_ax)
    cbar.set_label("Correlation", fontsize=9)

    plt.show()


if __name__ == "__main__":
    create_report()
