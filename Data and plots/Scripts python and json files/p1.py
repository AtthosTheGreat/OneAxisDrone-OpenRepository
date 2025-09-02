import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import math
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid")

class JSONMultiErrorComparer:
    def __init__(self, file1: str, file2: str, label1="Set1", label2="Set2"):
        self.label1 = label1
        self.label2 = label2
        self.users_data1 = self._load_users_data(file1)
        self.users_data2 = self._load_users_data(file2)
        self.common_users = sorted(set(self.users_data1.keys()) & set(self.users_data2.keys()))

    def _load_users_data(self, file_path):
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        users_data = {}
        for record in raw_data:
            username = record.get("username", "Unknown")
            data_block = record.get("data", {})

            users_data[username] = {
                "error_values": data_block.get("ErrorValues", []),
                # câmpul cu timpul jucat efectiv, așa cum vine din JSON
                "play_time_seconds": data_block.get("PlayTimeSeconds", None)
            }
        return users_data

    def _best_segment_stats(self, errors, window_size=100):
        if not errors or len(errors) < window_size:
            return (None, None, None, None, None)

        best_start_index = 0
        min_mean_abs = float('inf')
        best_mean = None
        best_segment = None

        for i in range(len(errors) - window_size + 1):
            segment = errors[i:i + window_size]
            mean_abs = np.mean(np.abs(segment))
            if mean_abs < min_mean_abs:
                min_mean_abs = mean_abs
                best_mean = float(np.mean(segment))
                best_segment = segment
                best_start_index = i

        seg_min = float(np.min(best_segment))
        seg_max = float(np.max(best_segment))
        start_seconds_from_index = best_start_index / 10.0
        return (seg_min, seg_max, best_mean, best_start_index, start_seconds_from_index)

    def build_summary_table(self, window_size=20, save_csv_path=None):
        rows = []
        for user in self.common_users:
            errors_before = self.users_data1[user].get("error_values", [])
            errors_after  = self.users_data2[user].get("error_values", [])

            b_min, b_max, b_mean, b_start_idx, b_start_sec_from_idx = self._best_segment_stats(errors_before, window_size)
            a_min, a_max, a_mean, a_start_idx, a_start_sec_from_idx = self._best_segment_stats(errors_after,  window_size)

            # timpul jucat real din JSON
            seconds_before_real = self.users_data1[user].get("play_time_seconds", None)
            seconds_after_real  = self.users_data2[user].get("play_time_seconds", None)

            rows.append({
                "User": user,

                "Min before": b_min,
                "Max before": b_max,
                "Segmentation Mean before": b_mean,
                "Start segmentation before": b_start_idx,
                "Start segmentation seconds before": b_start_sec_from_idx,
                "seconds before": seconds_before_real,

                "Min after": a_min,
                "Max after": a_max,
                "Segmentation Mean after": a_mean,
                "Start segmentation after": a_start_idx,
                "Start segmentation seconds after": a_start_sec_from_idx,
                "seconds after": seconds_after_real,
            })

        df = pd.DataFrame(rows)

        # rotunjire pt. afișare mai curată (lăsăm indecșii întregi)
        float_cols = [
            "Min before", "Max before", "Segmentation Mean before",
            "Start segmentation seconds before",
            "Min after", "Max after", "Segmentation Mean after",
            "Start segmentation seconds after"
        ]
        for c in float_cols:
            if c in df.columns:
                df[c] = df[c].astype(float).round(3)

        if save_csv_path:
            df.to_csv(save_csv_path, index=False)

        return df

    def plot_all_users_side_by_side_batched(self, subplots_per_fig=6, window_size=20):
        # Folosim doar utilizatorii comuni
        all_users = sorted(set(self.users_data1.keys()).intersection(set(self.users_data2.keys())))
        entries = []

        for user in all_users:
            entries.append((user, self.label1, self.users_data1[user].get("error_values", []), "tab:blue"),)
            entries.append((user, self.label2, self.users_data2[user].get("error_values", []), "tab:orange"))

        total = len(entries)
        num_figs = math.ceil(total / subplots_per_fig)

        for fig_idx in range(num_figs):
            start = fig_idx * subplots_per_fig
            end = min((fig_idx + 1) * subplots_per_fig, total)
            batch = entries[start:end]

            num_subplots = len(batch)
            num_cols = 2
            num_rows = math.ceil(num_subplots / num_cols)
            
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 4 * num_rows), sharex=False, sharey=False)
            axes = axes.flatten() if num_subplots > 1 else [axes]
            #---------------------------------------------------
            y_min_fixed = -70
            y_max_fixed = 70
            #---------------------------------------------------
            for ax, (user, label, errors, color) in zip(axes, batch):
                if not errors:
                    ax.set_title(f"{label} - {user} (no data)")
                    ax.axis("off")
                    continue

                dfp = pd.DataFrame({"Index": range(len(errors)), "ErrorValue": errors})
                sns.lineplot(x="Index", y="ErrorValue", data=dfp, ax=ax, color=color)

                # -- Segment optim (cele 20 de valori cele mai apropiate de 0) --
                seg_min, seg_max, seg_mean, start_idx, start_sec_from_idx = self._best_segment_stats(errors, window_size)

                if start_idx is not None:
                    ax.axhline(y=seg_mean, color='red', linestyle='--',
                               label=f'Segment Mean: {seg_mean:.3f}°')
                    ax.axhline(y=seg_min, color='green', linestyle='--',
                               label=f'Min: {seg_min:.3f}°')
                    ax.axhline(y=seg_max, color='purple', linestyle='--',
                               label=f'Max: {seg_max:.3f}°')
                    ax.axvline(x=start_idx, color='black', linestyle='-.', linewidth=1.5,
                               label=f'Start segment: ~{start_sec_from_idx:.1f}s')

                ax.set_title(f"{user} - {label}")
                ax.set_ylabel("Error Value (°)")
                ax.grid(True)
                #--------------------------------------
                ax.set_xlabel("Time (s)")
                ax.set_ylim(y_min_fixed, y_max_fixed)
                ax.set_xlim(0, 6000)
                ax.set_xticks(np.arange(0, 6001, 1000))  # poziții reale (0, 1000, 2000,...)
                ax.set_xticklabels([str(x // 10) for x in range(0, 6001, 1000)])  # afișează ca 0, 10, 20,...
                #--------------------------------------
                ax.legend(fontsize=8)

            for ax in axes[num_subplots:]:
                ax.axis("off")

            plt.tight_layout()
            plt.show()


# ----------------- EXEMPLE DE UTILIZARE -----------------
# Ajustează numele fișierelor după caz
comparer = JSONMultiErrorComparer("All_B.json", "All_A.json", label1="Before", label2="After")

# 1) Plotează ca înainte
comparer.plot_all_users_side_by_side_batched(window_size=20)

# 2) Generează TABELUL cerut (și opțional salvează CSV)
summary_df = comparer.build_summary_table(window_size=20, save_csv_path="summary_before_after.csv")
print(summary_df)
