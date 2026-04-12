"""Visualization utilities for multilingual roleplay analyses."""

import json
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.base.base_cultural_map_visualizer import BaseCulturalMapVisualizer


class MultilingualRoleplayVisualizer(BaseCulturalMapVisualizer):
    """Visualizer for multilingual roleplay outputs and diagnostics."""
    
    def __init__(self, data_path: str = "data", results_path: str = "results"):
        """Initialize output paths, palettes, and plotting defaults."""
        super().__init__(data_path=data_path, results_path=results_path)
        
        self.roleplay_results_path = self.results_path / "roleplay_ml_dashboard"
        self.roleplay_results_path.mkdir(parents=True, exist_ok=True)
        print(f"Visualization output directory: {self.roleplay_results_path}")
        
        self.REAL_COUNTRY_SIZE = 50
        self.REAL_COUNTRY_ALPHA = 0.7
        self.ROLEPLAY_SIZE = 100
        self.ROLEPLAY_ALPHA = 0.8
        
        self.PLOTLY_REAL_SIZE = 8
        self.PLOTLY_REAL_OPACITY = 0.7
        self.PLOTLY_ROLEPLAY_SIZE = 12
        self.PLOTLY_ROLEPLAY_OPACITY = 0.8
        
        self.language_colors = {
            'zh-cn': '#FF6B6B',
            'ru': '#4ECDC4',
            'es-la': '#45B7D1',
            'ar': '#FFA07A'
        }
        
        self.extended_colors.update(self.language_colors)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        sns.set_style("whitegrid")
    
    def load_data(self) -> pd.DataFrame:
        """Load the default PCA result table for visualization."""
        return self.load_pca_results()

    @staticmethod
    def _ensure_coordinate_aliases(data: pd.DataFrame) -> pd.DataFrame:
        """Provide legacy PC1/PC2 aliases for released rescaled-coordinate tables."""
        data = data.copy()
        if 'PC1' not in data.columns and 'PC1_rescaled' in data.columns:
            data['PC1'] = data['PC1_rescaled']
        if 'PC2' not in data.columns and 'PC2_rescaled' in data.columns:
            data['PC2'] = data['PC2_rescaled']
        if 'model' not in data.columns and 'model_name' in data.columns:
            data['model'] = data['model_name']
        if 'country' not in data.columns and 'country_code_clean' in data.columns:
            data['country'] = data['country_code_clean']
        return data

    @staticmethod
    def _multilingual_rows(df: pd.DataFrame) -> pd.DataFrame:
        """Return the model-generated subset used for roleplay diagnostics."""
        if 'model' in df.columns:
            return df[df['model'].notna()].copy()
        return df.copy()
    
    def load_pca_results(self, results_file: str = None) -> pd.DataFrame:
        """Load multilingual roleplay PCA results from the standard output path."""
        if results_file is None:
            roleplay_dir = self.data_path / "llm_pca" / "multilingual"
            
            data_paths = [
                roleplay_dir / "roleplay_ml_pca_entity_scores_latest.pkl",
                *sorted(roleplay_dir.glob("roleplay_ml_pca_entity_scores_*.pkl"), 
                       key=lambda x: x.stat().st_mtime, reverse=True),
                *sorted(roleplay_dir.glob("multilingual_pca_results_*.pkl"),
                       key=lambda x: x.stat().st_mtime, reverse=True)
            ]
            
            for data_path in data_paths:
                if data_path.exists():
                    data = self._ensure_coordinate_aliases(pd.read_pickle(data_path))
                    print(f"Loaded multilingual roleplay data: {data_path.name} {data.shape}")
                    return data
            
            raise FileNotFoundError(
                f"Could not find multilingual roleplay PCA data.\n"
                f"Expected path: {roleplay_dir / 'roleplay_ml_pca_entity_scores_latest.pkl'}\n"
                f"Run the PCA stage before creating visualizations."
            )
        else:
            results_file = Path(results_file)
            print(f"Loading PCA results: {results_file}")
            return self._ensure_coordinate_aliases(pd.read_pickle(results_file))
    
    def load_analysis_results(self, analysis_file: str = None) -> Dict:
        """Load the language-analysis summary used by the dashboards."""
        if analysis_file is None:
            processed_dir = self.data_path / "llm_interviews" / "multilingual" / "processed"
            analysis_file = processed_dir / "roleplay_ml_language_analysis_latest.json"
            
            if not analysis_file.exists():
                print("Language-analysis file not found; using an empty summary")
                return {'language_statistics': {}, 'language_differences': {}}
        else:
            analysis_file = Path(analysis_file)
        
        print(f"Loaded language-analysis summary: {analysis_file.name}")
        
        try:
            with open(analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Could not load language-analysis summary: {e}")
            return {'language_statistics': {}, 'language_differences': {}}
    

    def prepare_visualization_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split visualization inputs into IVS benchmark and roleplay subsets."""
        if 'data_source' in data.columns:
            real_countries = data[data['data_source'] == 'IVS'].copy()
            multilingual_data = data[data['data_source'] == 'Multilingual'].copy()
        else:
            real_countries = data[~data.get('model', '').notna()].copy()
            multilingual_data = data[data.get('model', '').notna()].copy()
        
        if len(real_countries) > 0:
            print(f"   IVS rows before aggregation: {len(real_countries)}")
            group_col = 'Country' if 'Country' in real_countries.columns else 'country_code'
            if group_col in real_countries.columns:
                agg_dict = {
                    'PC1_rescaled': 'mean',
                    'PC2_rescaled': 'mean',
                    'data_source': 'first'
                }
                if 'Cultural Region' in real_countries.columns:
                    agg_dict['Cultural Region'] = 'first'
                
                real_countries = real_countries.groupby(group_col).agg(agg_dict).reset_index()
                print(f"   IVS entities after aggregation: {len(real_countries)}")
        
        print("Prepared visualization inputs:")
        print(f"   IVS benchmark entities: {len(real_countries)}")
        print(f"   multilingual roleplay entities: {len(multilingual_data)}")
        
        return real_countries, multilingual_data


    def create_interactive_cultural_map(self, df: pd.DataFrame, suffix: str = None) -> str:
        """Create an interactive cultural map with IVS and roleplay points."""
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Creating interactive cultural map...")
        
        real_countries, multilingual_data = self.prepare_visualization_data(df)
        
        fig = go.Figure()
        
        if not real_countries.empty and 'PC1_rescaled' in real_countries.columns:
            if 'Cultural Region' in real_countries.columns:
                for region in real_countries['Cultural Region'].unique():
                    if pd.isna(region):
                        continue
                    subset = real_countries[real_countries['Cultural Region'] == region]
                    color = self.cultural_region_colors.get(region, '#cccccc')
                    
                    fig.add_trace(go.Scatter(
                        x=subset['PC1_rescaled'],
                        y=subset['PC2_rescaled'],
                        mode='markers',
                        name=f'{region} (Real Countries)',
                        marker=dict(
                            color=color,
                            size=self.PLOTLY_REAL_SIZE,
                            opacity=self.PLOTLY_REAL_OPACITY
                        ),
                        hovertemplate=(
                            '<b>%{text}</b><br>' +
                            f'Cultural Region: {region}<br>' +
                            'Type: Real Country<br>' +
                            'PC1: %{x:.3f}<br>' +
                            'PC2: %{y:.3f}<br>' +
                            '<extra></extra>'
                        ),
                        text=subset.get('Country', subset.get('country', '')),
                        showlegend=True,
                        legendgroup='IVS'
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=real_countries['PC1_rescaled'],
                    y=real_countries['PC2_rescaled'],
                    mode='markers',
                    name='Real Countries',
                    marker=dict(
                        color='#cccccc',
                        size=8,
                        opacity=0.7
                    ),
                    hovertemplate=(
                        '<b>%{text}</b><br>' +
                        'Type: Real Country<br>' +
                        'PC1: %{x:.3f}<br>' +
                        'PC2: %{y:.3f}<br>' +
                        '<extra></extra>'
                    ),
                    text=real_countries.get('Country', real_countries.get('country', '')),
                    showlegend=True
                ))
        
        if not multilingual_data.empty:
            valid_ml = multilingual_data.dropna(subset=['PC1', 'PC2'])
            
            if 'language' in valid_ml.columns:
                for language in valid_ml['language'].unique():
                    subset = valid_ml[valid_ml['language'] == language]
                    color = self.language_colors.get(language, '#999999')
                    
                    fig.add_trace(go.Scatter(
                        x=subset['PC1'],
                        y=subset['PC2'],
                        mode='markers',
                        name=f'Language: {language}',
                        marker=dict(
                            color=color,
                            size=self.PLOTLY_ROLEPLAY_SIZE,
                            opacity=self.PLOTLY_ROLEPLAY_OPACITY,
                            line=dict(width=1, color='white')
                        ),
                        hovertemplate=(
                            '<b>%{text}</b><br>' +
                            f'Language: {language}<br>' +
                            'Model: %{customdata[0]}<br>' +
                            'Country: %{customdata[1]}<br>' +
                            'PC1: %{x:.3f}<br>' +
                            'PC2: %{y:.3f}<br>' +
                            '<extra></extra>'
                        ),
                        text=subset.apply(lambda r: f"{r.get('model', 'Unknown')}-{r.get('country', 'Unknown')}", axis=1),
                        customdata=subset[['model', 'country']].values if 'model' in subset.columns and 'country' in subset.columns else None,
                        showlegend=True,
                        legendgroup='Multilingual'
                    ))
        
        fig.update_layout(
            title="Inglehart-Welzel Cultural Map: Stage 3 (Multilingual Roleplay with IVS Real Data)",
            xaxis_title="Survival vs. Self-Expression Values (PC1)",
            yaxis_title="Traditional vs. Secular-Rational Values (PC2)",
            hovermode='closest',
            template='plotly_white',
            width=1200,
            height=800,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        output_file = self.roleplay_results_path / f"multilingual_cultural_map_{suffix}.html"
        fig.write_html(str(output_file))
        print(f"Interactive cultural map saved: {output_file.name}")
        
        return str(output_file)

    def create_language_comparison_dashboard(self, df: pd.DataFrame, analysis_results: Dict, suffix: str = None) -> str:
        """Create a dashboard comparing languages across summary views."""
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Creating language comparison dashboard...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PC1 distribution', 'PC2 distribution', 'Language-distance heatmap', 'Model-language matrix'),
            specs=[[{"type": "box"}, {"type": "box"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        valid_df = self._multilingual_rows(df).dropna(subset=['PC1', 'PC2'])
        for language in valid_df['language'].unique():
            lang_data = valid_df[valid_df['language'] == language]
            fig.add_trace(
                go.Box(y=lang_data['PC1'], name=language, 
                      marker_color=self.language_colors.get(language, 'gray')),
                row=1, col=1
            )
        
        for language in valid_df['language'].unique():
            lang_data = valid_df[valid_df['language'] == language]
            fig.add_trace(
                go.Box(y=lang_data['PC2'], name=language, 
                      marker_color=self.language_colors.get(language, 'gray'),
                      showlegend=False),
                row=1, col=2
            )
        
        if 'language_statistics' in analysis_results:
            lang_stats = analysis_results['language_statistics']
            languages = list(lang_stats.keys())
            
            distance_matrix = np.zeros((len(languages), len(languages)))
            for i, lang1 in enumerate(languages):
                for j, lang2 in enumerate(languages):
                    if i != j and not np.isnan(lang_stats[lang1]['pc1_mean']) and not np.isnan(lang_stats[lang2]['pc1_mean']):
                        pc1_diff = abs(lang_stats[lang1]['pc1_mean'] - lang_stats[lang2]['pc1_mean'])
                        pc2_diff = abs(lang_stats[lang1]['pc2_mean'] - lang_stats[lang2]['pc2_mean'])
                        distance_matrix[i, j] = np.sqrt(pc1_diff**2 + pc2_diff**2)
            
            fig.add_trace(
                go.Heatmap(z=distance_matrix, x=languages, y=languages,
                          colorscale='Viridis', showscale=True),
                row=2, col=1
            )
        
        for model in valid_df['model'].unique():
            model_data = valid_df[valid_df['model'] == model]
            fig.add_trace(
                go.Scatter(x=model_data['PC1'], y=model_data['PC2'],
                          mode='markers', name=model.split('/')[-1],
                          marker=dict(size=8, opacity=0.6),
                          showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Multilingual roleplay analysis dashboard',
            height=800,
            showlegend=True
        )
        
        output_file = self.roleplay_results_path / f'multilingual_comparison_dashboard_{suffix}.html'
        fig.write_html(str(output_file))
        
        print(f"Language comparison dashboard saved: {output_file}")
        return str(output_file)
    
    def create_model_performance_analysis(self, df: pd.DataFrame, suffix: str = None) -> str:
        """Create a dashboard summarizing model-level performance."""
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Creating model performance analysis...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Model success rate',
                'Model-language heatmap',
                'Model PCA distribution',
                'Language coverage'
            ),
            specs=[[{"type": "bar"}, {"type": "heatmap"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        model_df = self._multilingual_rows(df)

        if 'success_rate' in model_df.columns:
            model_success = model_df.groupby('model')['success_rate'].mean().sort_values(ascending=False)
            
            fig.add_trace(
                go.Bar(x=model_success.index, y=model_success.values,
                      name='Success rate', marker_color='lightblue'),
                row=1, col=1
            )
        
        model_lang_matrix = model_df.groupby(['model', 'language']).size().unstack(fill_value=0)
        
        fig.add_trace(
            go.Heatmap(z=model_lang_matrix.values,
                      x=model_lang_matrix.columns,
                      y=model_lang_matrix.index,
                      colorscale='Blues'),
            row=1, col=2
        )
        
        valid_df = model_df.dropna(subset=['PC1', 'PC2'])
        for model in valid_df['model'].unique():
            model_data = valid_df[valid_df['model'] == model]
            fig.add_trace(
                go.Scatter(x=model_data['PC1'], y=model_data['PC2'],
                          mode='markers', name=model.split('/')[-1],
                          marker=dict(size=8, 
                                    color=self.llm_model_colors.get(model, 'gray'),
                                    opacity=0.6)),
                row=2, col=1
            )
        
        lang_coverage = model_df.groupby('language').size()
        
        fig.add_trace(
            go.Bar(x=lang_coverage.index, y=lang_coverage.values,
                  name='Sample count', marker_color='lightgreen'),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Model performance analysis',
            height=800,
            showlegend=True
        )
        
        output_file = self.roleplay_results_path / f'model_performance_analysis_{suffix}.html'
        fig.write_html(str(output_file))
        
        print(f"Model performance analysis saved: {output_file}")
        return str(output_file)
    
    def create_static_summary_plots(self, df: pd.DataFrame, analysis_results: Dict, suffix: str = None):
        """Create a compact static summary figure."""
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Creating static summary figure...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        model_df = self._multilingual_rows(df)
        valid_df = model_df.dropna(subset=['PC1', 'PC2'])
        for language in valid_df['language'].unique():
            lang_data = valid_df[valid_df['language'] == language]
            ax1.scatter(lang_data['PC1'], lang_data['PC2'],
                       c=self.language_colors.get(language, 'gray'),
                       label=language, s=100, alpha=0.7)

        ax1.set_xlabel('PC1 (Survival vs. Self-expression)')
        ax1.set_ylabel('PC2 (Traditional vs. Secular-rational)')
        ax1.set_title('Multilingual cultural map')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if 'language_statistics' in analysis_results:
            lang_stats = analysis_results['language_statistics']
            languages = []
            pc1_means = []
            pc1_stds = []
            
            for lang, stats in lang_stats.items():
                if not np.isnan(stats['pc1_mean']):
                    languages.append(lang)
                    pc1_means.append(stats['pc1_mean'])
                    pc1_stds.append(stats['pc1_std'])
            
            if languages:
                ax2.bar(languages, pc1_means, yerr=pc1_stds, capsize=5,
                       color=[self.language_colors.get(lang, 'gray') for lang in languages],
                       alpha=0.7)
                ax2.set_ylabel('Mean PC1')
                ax2.set_title('Mean PC1 by language')
                ax2.tick_params(axis='x', rotation=45)
        
        for model in valid_df['model'].unique():
            model_data = valid_df[valid_df['model'] == model]
            ax3.scatter(model_data['PC1'], model_data['PC2'],
                       label=model.split('/')[-1], s=60, alpha=0.6)
        
        ax3.set_xlabel('PC1 (Survival vs. Self-expression)')
        ax3.set_ylabel('PC2 (Traditional vs. Secular-rational)')
        ax3.set_title('Model distribution')
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        stats_data = {
            'Languages': model_df['language'].nunique(),
            'Models': model_df['model'].nunique(),
            'Countries': model_df['country'].nunique(),
            'Rows': len(model_df)
        }
        
        ax4.bar(stats_data.keys(), stats_data.values(), 
               color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
        ax4.set_ylabel('Count')
        ax4.set_title('Dataset summary')
        
        plt.tight_layout()
        plt.savefig(self.roleplay_results_path / f'multilingual_summary_{suffix}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Static summary figure saved")
    
    def generate_analysis_report(self, df: pd.DataFrame, analysis_results: Dict, suffix: str = None) -> str:
        """Generate a lightweight JSON analysis report."""
        if suffix is None:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("Generating analysis report...")
        
        model_df = self._multilingual_rows(df)

        report = {
            "title": "Multilingual roleplay analysis report",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_samples": len(model_df),
                "languages": model_df['language'].nunique(),
                "models": model_df['model'].nunique(),
                "countries": model_df['country'].nunique(),
                "valid_pca_samples": len(model_df.dropna(subset=['PC1', 'PC2']))
            },
            "language_analysis": {},
            "model_analysis": {},
            "key_findings": [],
            "recommendations": []
        }
        
        for language in model_df['language'].unique():
            lang_data = model_df[model_df['language'] == language]
            valid_lang_data = lang_data.dropna(subset=['PC1', 'PC2'])
            
            report["language_analysis"][language] = {
                "sample_count": len(lang_data),
                "valid_samples": len(valid_lang_data),
                "countries": lang_data['country'].unique().tolist(),
                "pc1_mean": valid_lang_data['PC1'].mean() if len(valid_lang_data) > 0 else np.nan,
                "pc2_mean": valid_lang_data['PC2'].mean() if len(valid_lang_data) > 0 else np.nan,
                "success_rate": lang_data['success_rate'].mean() if 'success_rate' in lang_data else np.nan
            }
        
        for model in model_df['model'].unique():
            model_data = model_df[model_df['model'] == model]
            valid_model_data = model_data.dropna(subset=['PC1', 'PC2'])
            
            report["model_analysis"][model] = {
                "sample_count": len(model_data),
                "valid_samples": len(valid_model_data),
                "languages": model_data['language'].unique().tolist(),
                "success_rate": model_data['success_rate'].mean() if 'success_rate' in model_data else np.nan
            }
        
        valid_df = model_df.dropna(subset=['PC1', 'PC2'])
        if len(valid_df) > 0:
            if 'language_differences' in analysis_results:
                lang_diffs = analysis_results.get('language_differences', {})
                if lang_diffs:
                    max_diff = max(lang_diffs.values(), key=lambda x: x['euclidean_distance'])
                    max_diff_pair = max(lang_diffs.keys(), key=lambda x: lang_diffs[x]['euclidean_distance'])
                    
                    report["key_findings"].append(
                        f"Largest cross-language distance: {max_diff_pair} (distance: {max_diff['euclidean_distance']:.3f})"
                    )
            
            pc1_range = valid_df['PC1'].max() - valid_df['PC1'].min()
            pc2_range = valid_df['PC2'].max() - valid_df['PC2'].min()
            
            report["key_findings"].extend([
                f"PC1 range: {pc1_range:.3f}",
                f"PC2 range: {pc2_range:.3f}",
                f"Data completeness: {len(valid_df)/len(df)*100:.1f}%"
            ])
        
        report["recommendations"] = [
            "Prioritize language pairs with large cultural distances for follow-up analysis.",
            "Review prompt design for model-language combinations with low success rates.",
            "Increase sample size where possible to improve statistical stability.",
            "Compare multilingual roleplay outputs directly against the English baseline."
        ]
        
        report_file = self.roleplay_results_path / f'multilingual_analysis_report_{suffix}.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"Analysis report saved: {report_file}")
        return str(report_file)
    
    def create_complete_visualization_suite(self, results_file: str = None, analysis_file: str = None) -> Dict:
        """Create the full multilingual roleplay visualization bundle."""
        print("=== Multilingual roleplay visualization suite ===")
        
        print("1. Loading PCA results...")
        df = self.load_pca_results(results_file)
        
        print("2. Loading analysis summary...")
        analysis_results = self.load_analysis_results(analysis_file)
        
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("3. Creating interactive cultural map...")
        cultural_map = self.create_interactive_cultural_map(df, suffix)
        
        print("4. Creating language comparison dashboard...")
        comparison_dashboard = self.create_language_comparison_dashboard(df, analysis_results, suffix)
        
        print("5. Creating model performance analysis...")
        model_analysis = self.create_model_performance_analysis(df, suffix)
        
        print("6. Creating static summary figure...")
        self.create_static_summary_plots(df, analysis_results, suffix)
        
        print("7. Generating analysis report...")
        analysis_report = self.generate_analysis_report(df, analysis_results, suffix)
        
        print("8. Building HTML index...")
        index_file = self._create_visualization_index(suffix, {
            'cultural_map': cultural_map,
            'comparison_dashboard': comparison_dashboard,
            'model_analysis': model_analysis,
            'analysis_report': analysis_report
        })
        
        print("\\n=== Visualization suite complete ===")
        print(f"Output directory: {self.roleplay_results_path}")
        print(f"Index page: {index_file}")
        
        return {
            'index_file': index_file,
            'cultural_map': cultural_map,
            'comparison_dashboard': comparison_dashboard,
            'model_analysis': model_analysis,
            'analysis_report': analysis_report,
            'viz_directory': str(self.roleplay_results_path)
        }
    
    def _create_visualization_index(self, suffix: str, file_paths: Dict) -> str:
        """Create a small HTML index page for the generated assets."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Multilingual roleplay analysis</title>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                .card {{ 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 20px; 
                    margin: 20px 0; 
                    background-color: #f9f9f9;
                }}
                .link {{ 
                    display: inline-block; 
                    padding: 10px 20px; 
                    background-color: #007bff; 
                    color: white; 
                    text-decoration: none; 
                    border-radius: 5px; 
                    margin: 5px;
                }}
                .link:hover {{ background-color: #0056b3; }}
            </style>
        </head>
        <body>
            <h1>Multilingual roleplay analysis</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="card">
                <h2>Interactive views</h2>
                <a href="{Path(file_paths['cultural_map']).name}" class="link">Cultural map</a>
                <a href="{Path(file_paths['comparison_dashboard']).name}" class="link">Language dashboard</a>
                <a href="{Path(file_paths['model_analysis']).name}" class="link">Model analysis</a>
            </div>
            
            <div class="card">
                <h2>Static summary</h2>
                <img src="multilingual_summary_{suffix}.png" alt="Multilingual summary" style="max-width: 100%; height: auto;">
            </div>
            
            <div class="card">
                <h2>Analysis report</h2>
                <a href="{Path(file_paths['analysis_report']).name}" class="link">Open JSON report</a>
            </div>
            
            <div class="card">
                <h2>What this bundle contains</h2>
                <ul>
                    <li>An interactive cultural map for IVS and multilingual roleplay points.</li>
                    <li>A dashboard comparing language-specific distributions.</li>
                    <li>A model-focused performance view.</li>
                    <li>A lightweight JSON report summarizing the generated outputs.</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        index_file = self.roleplay_results_path / f'multilingual_visualization_index_{suffix}.html'
        with open(index_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(index_file)
    
    def apply_duplicate_coordinate_offset(self, coords_data: pd.DataFrame, 
                                         multilingual_data: pd.DataFrame,
                                         identifier_cols: list) -> pd.DataFrame:
        """
        Apply a deterministic offset to points that share the same coordinates.

        The offset is derived from a hash of the identifier fields so that
        repeated runs produce the same jittered positions.
        """
        coords_data = coords_data.copy()
        
        for idx in coords_data.index:
            current_x = round(coords_data.loc[idx, 'PC1_rescaled'], 6)
            current_y = round(coords_data.loc[idx, 'PC2_rescaled'], 6)
            
            # Detect duplicates in the full multilingual dataset.
            global_duplicates = multilingual_data[
                (multilingual_data['PC1_rescaled'].round(6) == current_x) & 
                (multilingual_data['PC2_rescaled'].round(6) == current_y)
            ]
            
            if len(global_duplicates) > 1:
                # Use a deterministic offset derived from the point identity.
                current_row = multilingual_data.loc[idx] if idx in multilingual_data.index else None
                
                if current_row is not None:
                    identifier = '_'.join(str(current_row[col]) for col in identifier_cols if col in current_row.index)
                    hash_val = hash(identifier) % 1000
                    
                    offset_factor = 0.15
                    angle = (hash_val * 137.5) % 360
                    
                    offset_x = offset_factor * np.cos(np.radians(angle))
                    offset_y = offset_factor * np.sin(np.radians(angle))
                    
                    coords_data.loc[idx, 'PC1_rescaled'] += offset_x
                    coords_data.loc[idx, 'PC2_rescaled'] += offset_y
        
        return coords_data
    
    def generate_cultural_map_static(self, entity_scores: pd.DataFrame, save_path: str):
        """
        Generate a static cultural map using matplotlib.
        """
        print("Generating static cultural map...")
        
        # Split benchmark and multilingual points.
        ivs_data = entity_scores[entity_scores['data_source'] == 'IVS'].copy()
        multilingual_data = entity_scores[entity_scores['data_source'] == 'Multilingual'].copy()
        
        # Aggregate IVS data at the country level to avoid repeated survey years.
        if len(ivs_data) > 0:
            print(f"  IVS rows before aggregation: {len(ivs_data)}")
            group_col = 'Country' if 'Country' in ivs_data.columns else 'country_code'
            ivs_data = ivs_data.groupby(group_col).agg({
                'PC1_rescaled': 'mean',
                'PC2_rescaled': 'mean',
                'Cultural Region': 'first',
                'data_source': 'first'
            }).reset_index()
            print(f"  IVS countries after aggregation: {len(ivs_data)}")
        
        plt.style.use('default')
        fig, ax = plt.subplots(1, 1, figsize=(18, 14))
        
        # Draw IVS countries in the background.
        if len(ivs_data) > 0 and 'Cultural Region' in ivs_data.columns:
            regions = ivs_data['Cultural Region'].dropna().unique()
            
            for region in regions:
                if region in self.cultural_region_colors:
                    region_data = ivs_data[ivs_data['Cultural Region'] == region]
                    ax.scatter(
                        region_data['PC1_rescaled'], 
                        region_data['PC2_rescaled'],
                        c=self.cultural_region_colors[region], 
                        alpha=0.7, 
                        s=80, 
                        label=f'{region} ({len(region_data)} countries)',
                        marker='o',
                        edgecolors='white',
                        linewidth=1
                    )
        
        # Draw multilingual roleplay points in the foreground.
        if len(multilingual_data) > 0 and 'model_name' in multilingual_data.columns and 'language' in multilingual_data.columns:
            print(f"  Multilingual rows: {len(multilingual_data)}")
            models = sorted(multilingual_data['model_name'].dropna().unique())
            languages = sorted(multilingual_data['language'].dropna().unique())
            print(f"  Models: {len(models)}, languages: {len(languages)}")
            
            language_markers = {'english': 'o', 'native': 'D'}
            
            for model in models:
                model_short = model.split('/')[-1] if '/' in model else model
                model_color = self.llm_model_colors.get(model, '#95A5A6')
                
                for lang in languages:
                    model_lang_data = multilingual_data[
                        (multilingual_data['model_name'] == model) & 
                        (multilingual_data['language'] == lang)
                    ]
                    
                    if len(model_lang_data) > 0:
                        alpha = 0.8 if lang == 'english' else 0.6
                        size = 60 if lang == 'english' else 40
                        
                        coords_data = self.apply_duplicate_coordinate_offset(
                            coords_data=model_lang_data[['PC1_rescaled', 'PC2_rescaled']].copy(),
                            multilingual_data=multilingual_data,
                            identifier_cols=['country_code', 'model_name', 'language']
                        )
                        
                        ax.scatter(
                            coords_data['PC1_rescaled'], 
                            coords_data['PC2_rescaled'],
                            c=model_color, 
                            alpha=alpha, 
                            s=size, 
                            label=f'{model_short} ({lang.title()})',
                            marker=language_markers.get(lang, 'o'),
                            edgecolors='black',
                            linewidth=0.8
                        )
        
        ax.set_xlabel('PC1: Survival vs Self-Expression Values', fontsize=16, fontweight='bold')
        ax.set_ylabel('PC2: Traditional vs Secular-Rational Values', fontsize=16, fontweight='bold')
        ax.set_title('Cultural Values Map: Multilingual LLM Roleplay vs Real Countries\n(Inglehart-Welzel Framework)', 
                     fontsize=20, fontweight='bold', pad=25)
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.4, linewidth=1)
        
        quadrant_style = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray')
        ax.text(0.02, 0.98, 'Self-Expression\n& Secular-Rational', 
                transform=ax.transAxes, fontsize=12, ha='left', va='top', bbox=quadrant_style)
        ax.text(0.02, 0.02, 'Survival\n& Secular-Rational', 
                transform=ax.transAxes, fontsize=12, ha='left', va='bottom', bbox=quadrant_style)
        ax.text(0.98, 0.98, 'Self-Expression\n& Traditional', 
                transform=ax.transAxes, fontsize=12, ha='right', va='top', bbox=quadrant_style)
        ax.text(0.98, 0.02, 'Survival\n& Traditional', 
                transform=ax.transAxes, fontsize=12, ha='right', va='bottom', bbox=quadrant_style)
        
        legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, 
                           title='Cultural Regions & Languages', title_fontsize=12,
                           frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Static cultural map saved: {save_path}")
    
    def generate_cultural_map_interactive(self, entity_scores: pd.DataFrame, save_path: str):
        """
        Generate an interactive cultural map as a Plotly HTML file.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly is not installed; skipping interactive cultural map.")
            return
        
        print("Generating interactive cultural map...")
        
        # Split benchmark and multilingual points.
        ivs_data = entity_scores[entity_scores['data_source'] == 'IVS'].copy()
        multilingual_data = entity_scores[entity_scores['data_source'] == 'Multilingual'].copy()
        
        # Aggregate IVS data at the country level.
        if len(ivs_data) > 0:
            print(f"  IVS rows before aggregation: {len(ivs_data)}")
            group_col = 'Country' if 'Country' in ivs_data.columns else 'country_code'
            ivs_data = ivs_data.groupby(group_col).agg({
                'PC1_rescaled': 'mean',
                'PC2_rescaled': 'mean',
                'Cultural Region': 'first',
                'data_source': 'first'
            }).reset_index()
            print(f"  IVS countries after aggregation: {len(ivs_data)}")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=0, color='rgba(0,0,0,0)'),
            name='<b>Cultural Regions (IVS benchmark)</b>',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add IVS benchmark countries.
        if len(ivs_data) > 0 and 'Cultural Region' in ivs_data.columns:
            regions = sorted(ivs_data['Cultural Region'].dropna().unique())
            
            for region in regions:
                if region in self.cultural_region_colors:
                    region_data = ivs_data[ivs_data['Cultural Region'] == region]
                    
                    hover_text = []
                    for _, row in region_data.iterrows():
                        country_name = row.get('Country', 'Unknown')
                        if pd.isna(country_name):
                            country_name = f'Country {row.get("country_code", "Unknown")}'
                        
                        hover_text.append(
                            f'<b>{country_name}</b><br>' +
                            f'Region: {region}<br>' +
                            f'PC1 (Survival↔Self-Expression): {row["PC1_rescaled"]:.2f}<br>' +
                            f'PC2 (Traditional↔Secular): {row["PC2_rescaled"]:.2f}<br>' +
                            f'Source: Real Country (IVS)'
                        )
                    
                    fig.add_trace(go.Scatter(
                        x=region_data['PC1_rescaled'],
                        y=region_data['PC2_rescaled'],
                        mode='markers',
                        marker=dict(
                            color=self.cultural_region_colors[region],
                            size=12,
                            opacity=0.8,
                            line=dict(width=1, color='white'),
                            symbol='circle'
                        ),
                        name=f'  • {region} ({len(region_data)})',
                        text=hover_text,
                        hovertemplate='%{text}<extra></extra>',
                    ))
        
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=0, color='rgba(0,0,0,0)'),
            name='<b>Multilingual LLM roleplay</b>',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Add multilingual roleplay points.
        if len(multilingual_data) > 0 and 'model_name' in multilingual_data.columns and 'language' in multilingual_data.columns:
            models = sorted(multilingual_data['model_name'].dropna().unique())
            languages = sorted(multilingual_data['language'].dropna().unique())
            
            language_symbols = {'english': 'circle', 'native': 'diamond'}
            
            for model in models:
                model_short = model.split('/')[-1] if '/' in model else model
                model_color = self.llm_model_colors.get(model, '#95A5A6')
                
                for lang in languages:
                    model_lang_data = multilingual_data[
                        (multilingual_data['model_name'] == model) & 
                        (multilingual_data['language'] == lang)
                    ]
                    
                    if len(model_lang_data) > 0:
                        hover_text = []
                        for _, row in model_lang_data.iterrows():
                            country_name = row.get('country_code', 'Unknown')
                            
                            hover_text.append(
                                f'<b>{country_name}</b><br>' +
                                f'Model: {model_short}<br>' +
                                f'Language: {lang.title()}<br>' +
                                f'PC1 (Survival↔Self-Expression): {row["PC1_rescaled"]:.2f}<br>' +
                                f'PC2 (Traditional↔Secular): {row["PC2_rescaled"]:.2f}<br>' +
                                f'Source: Multilingual LLM'
                            )
                        
                        opacity = 0.8 if lang == 'english' else 0.6
                        size = 8 if lang == 'english' else 6
                        
                        coords_data = self.apply_duplicate_coordinate_offset(
                            coords_data=model_lang_data[['PC1_rescaled', 'PC2_rescaled']].copy(),
                            multilingual_data=multilingual_data,
                            identifier_cols=['country_code', 'model_name', 'language']
                        )
                        
                        fig.add_trace(go.Scatter(
                            x=coords_data['PC1_rescaled'],
                            y=coords_data['PC2_rescaled'],
                            mode='markers',
                            marker=dict(
                                color=model_color,
                                size=size,
                                opacity=opacity,
                                symbol=language_symbols.get(lang, 'circle'),
                                line=dict(width=1, color='black')
                            ),
                            name=f'  {model_short} ({lang.title()}) ({len(model_lang_data)})',
                            text=hover_text,
                            hovertemplate='%{text}<extra></extra>',
                        ))
        
        fig.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
        fig.add_vline(x=0, line_dash='dash', line_color='gray', opacity=0.5)
        
        fig.update_layout(
            title=dict(
                text='<b>Cultural Values Map: Multilingual LLM Roleplay vs Real Countries</b><br>' +
                     '<sub>Inglehart-Welzel Framework • Independent Legend Control</sub>',
                x=0.5,
                font=dict(size=20)
            ),
            xaxis=dict(
                title='<b>PC1: Survival vs Self-Expression Values</b>',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='<b>PC2: Traditional vs Secular-Rational Values</b>',
                titlefont=dict(size=14),
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray'
            ),
            width=1500,
            height=900,
            showlegend=True,
            legend=dict(
                orientation='v',
                yanchor='top',
                y=1,
                xanchor='left',
                x=1.02,
                font=dict(size=11),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='gray',
                borderwidth=1,
                itemsizing='constant'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(r=300)
        )
        
        annotations = [
            dict(x=0.02, y=0.98, xref='paper', yref='paper',
                 text='<b>Self-Expression<br>& Secular-Rational</b>',
                 showarrow=False, font=dict(size=12), 
                 bgcolor='rgba(255,255,255,0.8)', bordercolor='gray'),
            dict(x=0.02, y=0.02, xref='paper', yref='paper',
                 text='<b>Survival<br>& Secular-Rational</b>',
                 showarrow=False, font=dict(size=12),
                 bgcolor='rgba(255,255,255,0.8)', bordercolor='gray'),
            dict(x=0.75, y=0.98, xref='paper', yref='paper',
                 text='<b>Self-Expression<br>& Traditional</b>',
                 showarrow=False, font=dict(size=12),
                 bgcolor='rgba(255,255,255,0.8)', bordercolor='gray'),
            dict(x=0.75, y=0.02, xref='paper', yref='paper',
                 text='<b>Survival<br>& Traditional</b>',
                 showarrow=False, font=dict(size=12),
                 bgcolor='rgba(255,255,255,0.8)', bordercolor='gray')
        ]
        
        fig.update_layout(annotations=annotations)
        
        fig.write_html(save_path)
        print(f"Interactive cultural map saved: {save_path}")
    
    def plot_language_distance_comparison(self, comparison_results: dict, save_path: str = None):
        """
        Plot average English-versus-native distance comparisons.
        """
        from datetime import datetime
        
        summary = comparison_results['summary']
        
        distances = []
        labels = []
        colors = []
        
        if summary['native_language_avg_distance']:
            distances.append(summary['native_language_avg_distance'])
            labels.append('Native Language')
            colors.append('#2E8B57')
        
        if summary['english_language_avg_distance']:
            distances.append(summary['english_language_avg_distance'])
            labels.append('English Language')
            colors.append('#4169E1')
        
        if not distances:
            print("Not enough data to generate the distance comparison plot.")
            return
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(labels, distances, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        for bar, distance in zip(bars, distances):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{distance:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.title('Language Effectiveness Comparison\n(Lower Distance = Better Performance)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Average Distance to Real Countries', fontsize=14, fontweight='bold')
        plt.xlabel('Language Type', fontsize=14, fontweight='bold')
        
        if summary['language_improvement']:
            improvement = summary['language_improvement']
            if improvement > 0:
                text = f'English is {improvement:.1f}% better'
                color = 'lightblue'
            else:
                text = f'Native is {-improvement:.1f}% better'
                color = 'lightgreen'
            
            plt.text(0.5, max(distances) * 0.8, text, 
                    transform=plt.gca().transAxes, ha='center', fontsize=12,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7))
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.roleplay_results_path / f"language_distance_comparison_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Distance comparison plot saved: {save_path}")
    
    def plot_country_level_language_comparison(self, comparison_results: dict, save_path: str = None):
        """
        Plot country-level English-versus-native distance comparisons.
        """
        from datetime import datetime
        
        country_details = comparison_results['country_details']
        
        countries = []
        native_distances = []
        english_distances = []
        
        for country, data in country_details.items():
            if data['avg_native_distance'] or data['avg_english_distance']:
                countries.append(country)
                native_distances.append(data['avg_native_distance'] or 0)
                english_distances.append(data['avg_english_distance'] or 0)
        
        if not countries:
            print("Not enough data to generate the country-level comparison plot.")
            return
        
        # Limit the plot to the first 15 countries for readability.
        if len(countries) > 15:
            countries = countries[:15]
            native_distances = native_distances[:15]
            english_distances = english_distances[:15]
        
        x = np.arange(len(countries))
        width = 0.35
        
        plt.figure(figsize=(16, 10))
        
        plt.bar(x - width/2, native_distances, width, label='Native Language', 
               color='#2E8B57', alpha=0.8, edgecolor='black', linewidth=0.5)
        plt.bar(x + width/2, english_distances, width, label='English Language', 
               color='#4169E1', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        plt.title('Country-Level Language Effectiveness Comparison', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Average Distance to Real Country', fontsize=14, fontweight='bold')
        plt.xlabel('Countries', fontsize=14, fontweight='bold')
        plt.xticks(x, countries, rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.roleplay_results_path / f"country_level_language_comparison_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Country-level comparison plot saved: {save_path}")
    
    def plot_interactive_language_comparison(self, comparison_results: dict, save_path: str = None):
        """
        Generate an interactive country-level language comparison chart.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("Plotly is not installed; skipping interactive language comparison.")
            return
        
        from datetime import datetime
        
        country_details = comparison_results['country_details']
        
        countries = []
        native_distances = []
        english_distances = []
        
        for country, data in country_details.items():
            countries.append(country)
            native_distances.append(data['avg_native_distance'])
            english_distances.append(data['avg_english_distance'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Native Language',
            x=countries,
            y=native_distances,
            marker_color='#2E8B57',
            opacity=0.8,
            hovertemplate='<b>%{x}</b><br>Native Language Distance: %{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            name='English Language',
            x=countries,
            y=english_distances,
            marker_color='#4169E1',
            opacity=0.8,
            hovertemplate='<b>%{x}</b><br>English Distance: %{y:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='<b>Interactive Language Effectiveness Comparison</b><br><sub>Lower Distance = Better Performance</sub>',
                x=0.5,
                font=dict(size=18)
            ),
            xaxis=dict(
                title='<b>Countries</b>',
                titlefont=dict(size=14),
                tickangle=45
            ),
            yaxis=dict(
                title='<b>Average Distance to Real Country</b>',
                titlefont=dict(size=14)
            ),
            barmode='group',
            width=1200,
            height=700,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.roleplay_results_path / f"interactive_language_comparison_{timestamp}.html"
        
        fig.write_html(save_path)
        print(f"Interactive language comparison saved: {save_path}")
    
    def plot_model_language_comparison(self, comparison_results: dict, save_path: str = None):
        """
        Plot model-specific English-versus-native comparisons.
        """
        from datetime import datetime
        
        model_analysis = comparison_results.get('model_specific_analysis', {})
        
        if not model_analysis:
            print("No model-specific analysis is available; skipping model comparison plot.")
            return
        
        models = []
        native_distances = []
        english_distances = []
        improvements = []
        
        for model_name, stats in model_analysis.items():
            if stats['native_avg_distance'] and stats['english_avg_distance']:
                models.append(model_name.split('/')[-1])
                native_distances.append(stats['native_avg_distance'])
                english_distances.append(stats['english_avg_distance'])
                improvements.append(stats['language_improvement'])
        
        if not models:
            print("Not enough data to generate the model comparison plot.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, native_distances, width, label='Native Language', 
                       color='#2E8B57', alpha=0.8, edgecolor='black')
        bars2 = ax1.bar(x + width/2, english_distances, width, label='English Language', 
                       color='#4169E1', alpha=0.8, edgecolor='black')
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        for bar in bars2:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Distance to Real Countries', fontsize=12, fontweight='bold')
        ax1.set_title('Model-Specific Language Performance Comparison\n(Lower Distance = Better Performance)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        colors = ['#2E8B57' if imp < 0 else '#4169E1' for imp in improvements]
        bars3 = ax2.bar(models, improvements, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, imp in zip(bars3, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.5 if height > 0 else -0.5),
                    f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top', fontsize=10)
        
        ax2.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Language Improvement (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Language Effectiveness Improvement by Model\n(Positive = English Better, Negative = Native Better)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.roleplay_results_path / f"model_specific_language_comparison_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Model comparison plot saved: {save_path}")
    
    def plot_interactive_model_comparison(self, model_analysis: dict, save_path: str = None):
        """
        Generate an interactive model comparison chart using Plotly subplots.
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Plotly is not installed; skipping interactive model comparison.")
            return
        
        from datetime import datetime
        
        models = []
        native_distances = []
        english_distances = []
        improvements = []
        native_counts = []
        english_counts = []
        
        for model_name, stats in model_analysis.items():
            if stats['native_avg_distance'] and stats['english_avg_distance']:
                models.append(model_name.split('/')[-1])
                native_distances.append(stats['native_avg_distance'])
                english_distances.append(stats['english_avg_distance'])
                improvements.append(stats['language_improvement'])
                native_counts.append(stats['native_count'])
                english_counts.append(stats['english_count'])
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distance Comparison by Model', 'Language Improvement by Model'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(
                name='Native Language',
                x=models,
                y=native_distances,
                marker_color='#2E8B57',
                opacity=0.8,
                hovertemplate='<b>%{x}</b><br>Native Distance: %{y:.3f}<br>Count: %{customdata}<extra></extra>',
                customdata=native_counts
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(
                name='English Language',
                x=models,
                y=english_distances,
                marker_color='#4169E1',
                opacity=0.8,
                hovertemplate='<b>%{x}</b><br>English Distance: %{y:.3f}<br>Count: %{customdata}<extra></extra>',
                customdata=english_counts
            ),
            row=1, col=1
        )
        
        colors = ['#2E8B57' if imp < 0 else '#4169E1' for imp in improvements]
        fig.add_trace(
            go.Bar(
                name='Language Improvement',
                x=models,
                y=improvements,
                marker_color=colors,
                opacity=0.8,
                hovertemplate='<b>%{x}</b><br>Improvement: %{y:.1f}%<br>' +
                             '<i>Positive = English Better<br>Negative = Native Better</i><extra></extra>',
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='<b>Model-Specific Language Performance Analysis</b>',
                x=0.5,
                font=dict(size=18)
            ),
            barmode='group',
            width=1400,
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Models", row=1, col=1, tickangle=45)
        fig.update_yaxes(title_text="Average Distance", row=1, col=1)
        fig.update_xaxes(title_text="Models", row=1, col=2, tickangle=45)
        fig.update_yaxes(title_text="Improvement (%)", row=1, col=2)
        
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=1, col=2)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.roleplay_results_path / f"interactive_model_language_comparison_{timestamp}.html"
        
        fig.write_html(save_path)
        print(f"Interactive model comparison saved: {save_path}")


def main():
    """Build the full multilingual roleplay visualization suite."""
    print("Multilingual roleplay visualizer")
    print("=" * 60)
    
    try:
        visualizer = MultilingualRoleplayVisualizer()
        
        result = visualizer.create_complete_visualization_suite()
        print("\\nVisualization suite created successfully.")
        print("\\nOpen the index page to browse the outputs:")
        print(f"  {result['index_file']}")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
