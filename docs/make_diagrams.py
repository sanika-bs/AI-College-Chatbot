import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT = os.path.dirname(os.path.abspath(__file__))
def flow(title, boxes, fname, cols):
    fig, ax = plt.subplots(figsize=(13, 3.6 if len(boxes) <= cols else 6)); ax.axis("off"); ax.set_title(title, fontsize=14, weight="bold")
    rows = -(-len(boxes) // cols); ax.set_xlim(0, cols); ax.set_ylim(0, rows); pos = []
    for i, t in enumerate(boxes):
        r, c = divmod(i, cols); x, y = c + .5, rows - r - .5; pos.append((x, y))
        ax.add_patch(FancyBboxPatch((x - .42, y - .3), .84, .6, boxstyle="round,pad=0.02", fc="#e8ebff", ec="#3b4fd8"))
        ax.text(x, y, t, ha="center", va="center", fontsize=9.5, wrap=True)
    for a, b in zip(pos, pos[1:]):
        if a[1] == b[1]: ax.annotate("", xy=(b[0] - .43, b[1]), xytext=(a[0] + .43, a[1]), arrowprops=dict(arrowstyle="->", color="#3b4fd8"))
        else: ax.annotate("", xy=(b[0], b[1] + .31), xytext=(a[0], a[1] - .31), arrowprops=dict(arrowstyle="->", color="#3b4fd8", connectionstyle="arc3,rad=0"))
    plt.tight_layout(); plt.savefig(os.path.join(OUT, fname), dpi=130); plt.close()
flow("System architecture", ["Browser UI\n(HTML/CSS/JS)", "Flask API\n/ /chat /health", "Chatbot engine\nchatbot.py", "Preprocessing\n+ saved TF-IDF", "Logistic Regression\n(saved model)", "Response map\n(responses.json)"], "architecture.png", 3)
flow("Chatbot workflow", ["Student\nquestion", "Preprocess\n(clean, tokenize,\nnormalise)", "TF-IDF\nvectorise", "Intent\nclassifier", "Confidence\n>= threshold?", "Mapped answer\nor fallback", "JSON -> UI\n(intent, confidence)"], "workflow.png", 4)
