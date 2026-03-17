import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def build_presentation(pdf_path="reports/presentation_pack.pdf"):
    os.makedirs("reports", exist_ok=True)
    figs = [
        "reports/confusion_matrix.png",
        "reports/snr_curve.png",
        "reports/snr_curve_per_class.png",
        "reports/snr_curve_impaired.png",
        "reports/snr_curve_per_class_impaired.png",
        "reports/figures/ser_curves.png",
        "reports/figures/snn_vs_ser.png",
    ]
    with PdfPages(pdf_path) as pdf:
        if os.path.exists("reports/results.txt"):
            txt = open("reports/results.txt").read()
            plt.figure(figsize=(8.5, 11))
            plt.axis("off")
            plt.text(0.05, 0.95, "Neuromorphic Cognitive Radio", fontsize=16, ha="left", va="top")
            plt.text(0.05, 0.9, "Summary", fontsize=12, ha="left", va="top")
            plt.text(0.05, 0.85, txt, fontsize=10, ha="left", va="top", wrap=True)
            pdf.savefig()
            plt.close()
        for fp in figs:
            if os.path.exists(fp):
                img = plt.imread(fp)
                plt.figure(figsize=(8.5, 6))
                plt.imshow(img)
                plt.axis("off")
                pdf.savefig()
                plt.close()
    return pdf_path

if __name__ == "__main__":
    print(build_presentation())
