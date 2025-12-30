import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA

class AcademicVisualizer:
    def __init__(self, output_dir="plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font="serif")

    def generate_plots(self, df, winner, synthons):
        print(f"[VIZ] Generating 4 Retrosynthetic Blueprints in /{self.output_dir}...")
        
        # Figure 1 Score Distribution
        self._plot_priority_distribution(df)
        
        # Figure 2 LATENT SPACE CLUSTERS (The missing plot)
        self._plot_pca_clusters(df)
        
        # Figure 3 Decision Heatmap
        self._plot_decision_heatmap(df)
        
        # Figure 5 Synthon Disconnection Map
        self._plot_retrosynthetic_map(winner, synthons)

    def _plot_priority_distribution(self, df):
        plt.figure(figsize=(8, 5))
        sns.histplot(df['score'], kde=True, color='#2c3e50', bins=15)
        plt.title("Fig 1: Retrosynthetic Priority Distribution", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Fig1_Scores.png", dpi=300)
        plt.close()

    def _plot_pca_clusters(self, df):
        """Fig 2: Visualizes the AI's internal 'Chemical Space' mapping."""
        # Extract embeddings (the 'vec' column)
        embeddings = np.stack(df['vec'].values)
        
        # Reduce to 2 dimensions for visualization
        pca = PCA(n_components=2)
        coords = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=df['score'], 
                            cmap='viridis', alpha=0.7, edgecolors='w')
        plt.colorbar(scatter, label='Retrosynthetic Priority')
        plt.title("Fig 2: Latent Space Clustering of Proteomic Targets", fontsize=12)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Fig2_PCA.png", dpi=300)
        plt.close()

    def _plot_decision_heatmap(self, df):
        plt.figure(figsize=(10, 4))
        top_5 = df.sort_values(by="score", ascending=False).head(5)
        plot_data = top_5[['score']].copy()
        plot_data.index = [f"ID: {id[:8]}" for id in top_5['id']]
        sns.heatmap(plot_data.T, annot=True, cmap="YlGnBu")
        plt.title("Fig 3: Retrosynthetic Target Decision Matrix", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Fig3_Heatmap.png", dpi=300)
        plt.close()

    def _plot_retrosynthetic_map(self, winner, synthons):
        plt.figure(figsize=(12, 4))
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        colors = sns.color_palette("flare", len(synthons))
        for i, s in enumerate(synthons):
            start = s['index']
            end = start + 15
            plt.plot([start, end], [0.5, 0.5], linewidth=10, color=colors[i], 
                     label=f"Synthon {i+1}")
        plt.title(f"Fig 4: Strategic Disconnection Map (Synthons)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/Fig4_Synthon_Map.png", dpi=300)
        plt.close()