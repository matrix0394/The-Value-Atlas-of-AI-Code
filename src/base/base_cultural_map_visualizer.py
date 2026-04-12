"""Shared plotting helpers for cultural-map visualizations."""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from sklearn.svm import SVC
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod


class BaseCulturalMapVisualizer(ABC):
    """Base class for cultural-map visualizations."""
    
    def __init__(self, data_path: str = "data", results_path: str = "results"):
        """Initialize shared data and output paths."""
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(exist_ok=True)
        
        # Shared region palette used across benchmark and LLM plots.
        self.cultural_region_colors = {
            'Orthodox Europe': '#0072b2',
            'Catholic Europe': '#e69f00',
            'Protestant Europe': '#d55e00',
            'English-Speaking': '#009e73',
            'Confucian': '#cc0000',
            'West & South Asia': '#f0e442',
            'African-Islamic': '#cc79a7',
            'Africa-Islamic': '#cc79a7',
            'Latin America': '#7f7f7f',
            'AI Model': '#ff1493',
            'Unknown': '#cccccc'
        }
        
        self.llm_model_colors = self._generate_llm_model_colors()
        
        self.extended_colors = self._generate_extended_colors()
    
    def _generate_llm_model_colors(self) -> Dict[str, str]:
        """Return a consistent color mapping for the LLMs used in the paper."""
        base_colors = {
            'gpt-4o': '#ff4444',
            'gpt-4o-mini': '#ff6666',
            'claude-3-7-sonnet': '#00bfff',
            'gemini-2.5-flash': '#4285f4',
            'gemini-2.5-pro': '#5a9fd4',
            'deepseek-chat': '#ff6b35',
            'kimi-k2': '#9966cc',
            'qwen3-1.7b': '#ffaa00',
            'openai/gpt-5.1': '#ff3333',
            'anthropic/claude-sonnet-4.5': '#4169e1',
            'google/gemini-3-pro-previewl': '#6eb5d4',
            'google/gemma-3-4b-it': '#87ceeb',
            'meta-llama/llama-3.3-70b-instruct': '#8b4513',
            'meta-llama/llama-3.2-3b-instruct': '#a0522d',
            'x-ai/grok-4.1-fast': '#00ffff',
            'deepseek/deepseek-chat-v3.1': '#ff8c5a',
            'qwen/qwen3-max': '#ffbb33',
            'qwen/qwq-32b-preview': '#ffcc55',
            'z-ai/glm-4.6': '#20b2aa',
            'mistralai/mistral-nemo': '#9370db',
            'mistralai/mistral-medium-3.1': '#ba55d3',
            'microsoft/phi-3-mini-128k-instruct': '#0078d4',
        }
        
        full_mapping = {}
        full_mapping.update(base_colors)
        
        # Common aliases point to the same family colors.
        aliases = {
            'gpt': base_colors['gpt-4o'],
            'GPT': base_colors['gpt-4o'],
            'gpt-4o': base_colors['gpt-4o'],
            'gpt-4o-mini': base_colors['gpt-4o-mini'],
            
            'claude': base_colors['claude-3-7-sonnet'],
            'Claude': base_colors['claude-3-7-sonnet'],
            
            'gemini': base_colors['gemini-2.5-flash'],
            'Gemini': base_colors['gemini-2.5-flash'],
            
            'llama': base_colors['meta-llama/llama-3.3-70b-instruct'],
            'LLaMA': base_colors['meta-llama/llama-3.3-70b-instruct'],
            
            'deepseek': base_colors['deepseek-chat'],
            'DeepSeek': base_colors['deepseek-chat'],
            
            'qwen': base_colors['qwen3-1.7b'],
            'QWen': base_colors['qwen3-1.7b'],
            
            'kimi': base_colors['kimi-k2'],
            'Kimi': base_colors['kimi-k2'],
            
            'mistral': base_colors['mistralai/mistral-nemo'],
            'Mistral': base_colors['mistralai/mistral-nemo'],
            
            'glm': base_colors['z-ai/glm-4.6'],
            'GLM': base_colors['z-ai/glm-4.6'],
            
            'grok': base_colors['x-ai/grok-4.1-fast'],
            'Grok': base_colors['x-ai/grok-4.1-fast'],
            
            'gemma': base_colors['google/gemma-3-4b-it'],
            'Gemma': base_colors['google/gemma-3-4b-it'],
            
            'phi': base_colors['microsoft/phi-3-mini-128k-instruct'],
            'Phi': base_colors['microsoft/phi-3-mini-128k-instruct'],
        }
        
        full_mapping.update(aliases)
        
        full_mapping['LLM'] = '#ff1493'
        full_mapping['Model'] = '#ff1493'
        
        return full_mapping
    
    def _generate_extended_colors(self) -> Dict[str, str]:
        """Combine region and model color mappings."""
        extended = {}
        extended.update(self.cultural_region_colors)
        extended.update(self.llm_model_colors)
        
        backup_colors = {
            'backup_1': '#2ca02c',
            'backup_2': '#bcbd22',
            'backup_3': '#17becf',
            'backup_4': '#8c564b',
            'backup_5': '#e377c2'
        }
        extended.update(backup_colors)
        
        return extended
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load the dataframe required by a concrete visualizer."""
        pass
    
    def get_color_for_region(self, region: str) -> str:
        """Return the color assigned to a region or region-like label."""
        if not region or pd.isna(region):
            return '#cccccc'
        
        region_str = str(region).strip()
        
        if region_str in self.extended_colors:
            return self.extended_colors[region_str]
        
        region_lower = region_str.lower()
        for key in self.llm_model_colors.keys():
            if key.lower() in region_lower or region_lower in key.lower():
                return self.llm_model_colors[key]
        
        return '#cccccc'
    
    def get_color_for_model(self, model_name: str) -> str:
        """Return the color assigned to an LLM model name."""
        if not model_name or pd.isna(model_name):
            return self.llm_model_colors.get('LLM', '#ff1493')
        
        model_str = str(model_name).strip()
        
        if model_str in self.llm_model_colors:
            return self.llm_model_colors[model_str]
        
        model_lower = model_str.lower()
        
        # Match families in a stable order to avoid false positives.
        model_patterns = [
            ('gpt-5', '#ff3333'),
            ('gpt-4o-mini', '#ff6666'),
            ('gpt-4o', '#ff4444'),
            ('gpt', '#ff4444'),
            ('claude-sonnet-4.5', '#4169e1'),
            ('claude-3-7', '#00bfff'),
            ('claude', '#00bfff'),
            ('gemini-3', '#6eb5d4'),
            ('gemini-2.5-pro', '#5a9fd4'),
            ('gemini-2.5', '#4285f4'),
            ('gemini', '#4285f4'),
            ('gemma', '#87ceeb'),
            ('llama-3.3', '#8b4513'),
            ('llama-3.2', '#a0522d'),
            ('llama', '#8b4513'),
            ('deepseek-v3.1', '#ff8c5a'),
            ('deepseek-chat', '#ff6b35'),
            ('deepseek', '#ff6b35'),
            ('qwq', '#ffcc55'),
            ('qwen3-max', '#ffbb33'),
            ('qwen3', '#ffaa00'),
            ('qwen', '#ffaa00'),
            ('kimi', '#9966cc'),
            ('glm', '#20b2aa'),
            ('mistral-medium', '#ba55d3'),
            ('mistral-nemo', '#9370db'),
            ('mistral', '#9370db'),
            ('grok', '#00ffff'),
            ('phi', '#0078d4'),
        ]
        
        for pattern, color in model_patterns:
            if pattern in model_lower:
                return color
        
        return self.llm_model_colors.get('LLM', '#ff1493')
    
    def plot_basic_cultural_map(self, data: pd.DataFrame, 
                               figsize: Tuple[int, int] = (14, 10),
                               title: str = "Inglehart-Welzel Cultural Map",
                               show_labels: bool = True,
                               save_path: Optional[str] = None) -> plt.Figure:
        """Plot a basic cultural map for the provided dataframe."""
        fig, ax = plt.subplots(figsize=figsize)
        
        for region in data['Cultural Region'].unique():
            if pd.isna(region):
                continue
                
            subset = data[data['Cultural Region'] == region]
            color = self.get_color_for_region(region)
            
            ax.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], 
                      label=region, color=color, s=50, alpha=0.7)
            
            if show_labels:
                for _, row in subset.iterrows():
                    label_text = self._get_point_label(row)
                    if label_text:
                        ax.text(row['PC1_rescaled'], row['PC2_rescaled'], 
                               label_text, color=color, fontsize=8, ha='center')
        
        ax.set_xlabel('Survival vs. Self-Expression Values')
        ax.set_ylabel('Traditional vs. Secular Values')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cultural map saved to: {save_path}")
        
        return fig
    
    def _get_point_label(self, row: pd.Series) -> str:
        """Return the point label for one row."""
        if 'Country' in row.index:
            return str(row['Country'])
        elif 'model_name' in row.index:
            return str(row['model_name'])
        else:
            return ""
    
    def plot_decision_boundary(self, data: pd.DataFrame,
                              target_column: str = 'Cultural Region',
                              figsize: Tuple[int, int] = (14, 10),
                              save_path: Optional[str] = None) -> plt.Figure:
        """Plot an SVM decision boundary over the cultural map."""
        fig, ax = plt.subplots(figsize=figsize)
        
        X = data[['PC1_rescaled', 'PC2_rescaled']].values
        y = data[target_column].values
        
        svm = SVC(kernel='rbf', C=1.0, gamma='scale')
        svm.fit(X, y)
        
        h = 0.1
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                            np.arange(y_min, y_max, h))
        
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set3)
        
        for region in data[target_column].unique():
            if pd.isna(region):
                continue
            subset = data[data[target_column] == region]
            color = self.get_color_for_region(region)
            ax.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], 
                      label=region, color=color, s=50, alpha=0.8)
        
        ax.set_xlabel('Survival vs. Self-Expression Values')
        ax.set_ylabel('Traditional vs. Secular Values')
        ax.set_title('Cultural Regions with Decision Boundaries')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Decision-boundary plot saved to: {save_path}")
        
        return fig
    
    def create_comparison_plot(self, data: pd.DataFrame,
                              group_column: str,
                              figsize: Tuple[int, int] = (16, 12),
                              save_path: Optional[str] = None) -> plt.Figure:
        """Create a four-panel comparison plot for grouped data."""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        ax1 = axes[0, 0]
        for group in data[group_column].unique():
            if pd.isna(group):
                continue
            subset = data[data[group_column] == group]
            color = self.get_color_for_region(str(group))
            ax1.scatter(subset['PC1_rescaled'], subset['PC2_rescaled'], 
                       label=group, color=color, alpha=0.7)
        ax1.set_title('Cultural Distribution')
        ax1.set_xlabel('PC1 (Survival vs. Self-Expression)')
        ax1.set_ylabel('PC2 (Traditional vs. Secular)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for group in data[group_column].unique():
            if pd.isna(group):
                continue
            subset = data[data[group_column] == group]
            ax2.hist(subset['PC1_rescaled'], alpha=0.6, label=group, bins=15)
        ax2.set_title('PC1 Distribution')
        ax2.set_xlabel('PC1 Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        ax3 = axes[1, 0]
        for group in data[group_column].unique():
            if pd.isna(group):
                continue
            subset = data[data[group_column] == group]
            ax3.hist(subset['PC2_rescaled'], alpha=0.6, label=group, bins=15)
        ax3.set_title('PC2 Distribution')
        ax3.set_xlabel('PC2 Score')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        ax4 = axes[1, 1]
        summary_data = data.groupby(group_column)[['PC1_rescaled', 'PC2_rescaled']].agg(['mean', 'std'])
        summary_data.plot(kind='bar', ax=ax4)
        ax4.set_title('Statistical Summary')
        ax4.set_ylabel('Score')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, 
                   subfolder: str = None) -> str:
        """Save a figure under the configured results directory."""
        if subfolder:
            save_dir = self.results_path / subfolder
            save_dir.mkdir(exist_ok=True)
        else:
            save_dir = self.results_path
        
        save_path = save_dir / filename
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
        return str(save_path)
