import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

from analysis.distribution import run_distribution
from analysis.sms_length import run_sms_length
from analysis.sms_word_count import run_sms_word_count
from analysis.sms_sentence_count import run_sms_sentence_count
from analysis.word_analysis_stop import run_word_stop_analysis
from analysis.correlation import compute_correlation_matrix, compute_correlation_by_label
import logging

logging.getLogger().setLevel(logging.INFO)

OUTPUT_DIR = Path(__file__).parent / "img"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_pie(x_labels, values, title, filename, colors=["#66b3ff", "#ff6666"], explode=(0.02, 0.02)):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=x_labels, autopct="%1.1f%%", colors=colors, explode=explode, startangle=90)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {OUTPUT_DIR / filename}")


def save_bar(x_labels, values, title, ylabel, filename, color="#3498db", dual=False, values2=None, labels=("Mean", "Median")):
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(x_labels))
    
    if dual and values2 is not None:
        width = 0.35
        ax.bar(x - width/2, values, width, label=labels[0], color="#3498db")
        ax.bar(x + width/2, values2, width, label=labels[1], color="#2ecc71")
        ax.legend()
    else:
        ax.bar(x, values, color=color)
    
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {OUTPUT_DIR / filename}")


def save_hbar(words, title, filename, color="#ff6666"):
    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(words))
    ax.barh(y_pos, range(len(words), 0, -1), color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel("Rank")
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {OUTPUT_DIR / filename}")


def save_heatmap(corr_matrix, title, filename, cmap="coolwarm"):
    short_names = {"text_length": "Length", "word_count": "Words", "sentence_count": "Sentences"}
    columns = [short_names.get(c, c) for c in corr_matrix.columns]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr_matrix.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)
    ax.set_title(title, fontsize=14, fontweight="bold")
    
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            val = corr_matrix.iloc[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=11, color=color, fontweight="bold")
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved: {OUTPUT_DIR / filename}")


def create_report_images():
    logging.info("Generating individual report images...")
    
    logging.info("Running analyses...")
    dist = run_distribution()
    length_stats = run_sms_length()
    word_counts = run_sms_word_count()
    sentence_counts = run_sms_sentence_count()
    word_results = run_word_stop_analysis(top_n=10)
    corr_overall = compute_correlation_matrix()
    corr_by_label = compute_correlation_by_label()
    
    logging.info("Creating charts...")
    
    save_pie(dist.index, dist.values, "Label Distribution", "01_label_distribution.png")
    
    save_bar(length_stats.index, length_stats["mean"], "Text Length by Label (Mean & Median)", "Characters", 
             "02_text_length_mean.png", "#3498db", dual=True, values2=length_stats["median"].values, labels=("Mean", "Median"))
    
    save_bar(word_counts.index, word_counts.values, "Word Count by Label (Median)", "Word Count", 
             "03_word_count.png", "#9b59b6")
    
    save_bar(sentence_counts.index, sentence_counts.values, "Sentence Count by Label (Median)", "Sentence Count", 
             "04_sentence_count.png", "#f39c12")
    
    save_hbar(word_results["spam"], "Top 10 Words (Spam)", "05_top_words_spam.png", "#ff6666")
    
    save_hbar(word_results["ham"], "Top 10 Words (Ham)", "06_top_words_ham.png", "#66b3ff")
    
    save_heatmap(corr_overall, "Correlation Matrix (Overall)", "07_correlation_overall.png")
    
    save_heatmap(corr_by_label.get("ham", corr_overall), "Correlation Matrix (Ham)", "08_correlation_ham.png")
    
    save_heatmap(corr_by_label.get("spam", corr_overall), "Correlation Matrix (Spam)", "09_correlation_spam.png")
    
    logging.info(f"All images saved to: {OUTPUT_DIR}")
    logging.info("Done!")


if __name__ == "__main__":
    create_report_images()
