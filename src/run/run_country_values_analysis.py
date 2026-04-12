#!/usr/bin/env python3
"""
Run the country-benchmark pipeline used in the paper.

This script covers three stages:
1. Process the IVS source file into the filtered benchmark tables.
2. Fit the PCA model and export country coordinates.
3. Render summary visualizations and coordinate files.
"""

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add the project root so local modules can be imported when the script is run directly.
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.country_values.data_processing import DataProcessor
from src.country_values.pca_analysis import CorePCAAnalyzer
from src.country_values.visualization import CulturalMapVisualizer


class CountryValuesAnalysisRunner:
    """Runner for the country-benchmark pipeline."""
    
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
        
        self.data_path = self.project_root / "data" / "country_values"
        self.results_path = self.project_root / "results" / "country_values"
        self.raw_data_path = self.project_root / "data" / "raw"
        
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Project root: {self.project_root}")
        print(f"Data directory: {self.data_path}")
        print(f"Results directory: {self.results_path}")
    
    def step1_data_processing(self):
        """Process the IVS source file and save the filtered benchmark tables."""
        print("\n" + "="*60)
        print("Step 1: Data processing")
        print("="*60)
        
        primary_ivs_file = self.raw_data_path / "Integrated_values_surveys_1981-2022.sav"
        legacy_ivs_file = self.data_path / "Integrated_values_surveys_1981-2022.sav"

        if primary_ivs_file.exists():
            ivs_file = primary_ivs_file
        elif legacy_ivs_file.exists():
            ivs_file = legacy_ivs_file
            print(f"Using compatibility path for raw IVS data: {ivs_file}")
        else:
            print(f"Missing raw IVS file: {primary_ivs_file}")
            print(f"Compatibility path also missing: {legacy_ivs_file}")
            print("Place the raw IVS file under data/raw/ or data/country_values/.")
            return False
        
        processor = DataProcessor(data_path=str(self.data_path))
        
        try:
            print("\nLoading IVS data...")
            ivs_df = processor.load_ivs_data(str(ivs_file))
            print(f"Loaded {len(ivs_df)} rows and {len(ivs_df.columns)} columns.")
            
            print("\nChecking country_codes.pkl...")
            config_country_path = self.project_root / "config" / "country" / "country_codes.pkl"
            if not config_country_path.exists():
                country_codes = processor.create_country_codes()
                print(f"Created country_codes.pkl with {len(country_codes)} countries.")
            else:
                print("country_codes.pkl already exists in config/country/.")
            
            print("\nFiltering benchmark data...")
            filtered_data = processor.get_filtered_data()
            print(f"Filtered dataset contains {len(filtered_data)} rows.")
            
            print("\nSaving processed data...")
            processor.save_data()
            print("Processed data saved.")
            
            print("\nDataset summary:")
            print(f"Raw data shape: {ivs_df.shape}")
            print(f"Filtered data shape: {filtered_data.shape}")
            unique_countries = sorted(filtered_data['country_code'].unique())
            print(f"Countries included: {len(unique_countries)}")
            print(f"Country codes: {unique_countries}")
            
            return True
            
        except Exception as e:
            print(f"Data processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step2_pca_analysis(self):
        """Fit the PCA model and export country coordinates."""
        print("\n" + "="*60)
        print("Step 2: PCA analysis")
        print("="*60)
        
        try:
            config_country_path = self.project_root / "config" / "country" / "country_codes.pkl"
            required_files = [
                self.data_path / "ivs_df.pkl",
                config_country_path,
                self.data_path / "valid_data.pkl"
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    print(f"Missing required file: {file_path}")
                    return False
            
            print("\nInitializing PCA analyzer...")
            analyzer = CorePCAAnalyzer(data_path=str(self.data_path))
            
            print("\nRunning PCA analysis...")
            country_scores = analyzer.run_full_analysis()
            
            print(f"\nPCA analysis completed for {len(country_scores)} countries.")
            
            if 'PC1_rescaled' in country_scores.columns:
                print(f"PC1 range: [{country_scores['PC1_rescaled'].min():.2f}, {country_scores['PC1_rescaled'].max():.2f}]")
            if 'PC2_rescaled' in country_scores.columns:
                print(f"PC2 range: [{country_scores['PC2_rescaled'].min():.2f}, {country_scores['PC2_rescaled'].max():.2f}]")
            if 'Cultural Region' in country_scores.columns:
                print("\nCultural-region counts:")
                print(country_scores['Cultural Region'].value_counts())
            
            return True
            
        except Exception as e:
            print(f"PCA analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step3_visualization(self):
        """Create benchmark figures and export summary files."""
        print("\n" + "="*60)
        print("Step 3: Visualization")
        print("="*60)
        
        try:
            country_scores_path = self.data_path / "country_scores_pca.pkl"
            if not country_scores_path.exists():
                print(f"Missing PCA results file: {country_scores_path}")
                return False
            
            print("\nInitializing visualizer...")
            visualizer = CulturalMapVisualizer(data_path=str(self.data_path))
            
            print("\nLoading country scores...")
            country_scores_pca = visualizer.load_country_scores()
            print(f"Loaded scores for {len(country_scores_pca)} countries.")
            
            print("\nRendering cultural map...")
            cultural_map_path = self.results_path / "cultural_map.png"
            visualizer.plot_cultural_map(save_path=str(cultural_map_path))
            print(f"Saved cultural map to: {cultural_map_path}")
            
            print("\nRendering decision-boundary plot...")
            decision_boundary_path = self.results_path / "decision_boundary.png"
            visualizer.plot_decision_boundary(save_path=str(decision_boundary_path))
            print(f"Saved decision-boundary plot to: {decision_boundary_path}")
            
            print("\nWriting summary statistics...")
            summary_stats = visualizer.create_summary_statistics()
            
            summary_path = self.results_path / "summary_statistics.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Country Values Analysis Summary\n")
                f.write("="*50 + "\n\n")
                f.write(f"Total countries: {len(country_scores_pca)}\n\n")
                
                if 'Cultural Region' in country_scores_pca.columns:
                    f.write("Countries by Cultural Region:\n")
                    region_counts = country_scores_pca['Cultural Region'].value_counts()
                    for region, count in region_counts.items():
                        f.write(f"  {region}: {count}\n")
                    f.write("\n")
                
                f.write("Principal Component Statistics:\n")
                f.write(f"PC1 range: [{country_scores_pca['PC1_rescaled'].min():.2f}, {country_scores_pca['PC1_rescaled'].max():.2f}]\n")
                f.write(f"PC2 range: [{country_scores_pca['PC2_rescaled'].min():.2f}, {country_scores_pca['PC2_rescaled'].max():.2f}]\n\n")
                
                f.write("Detailed Statistics:\n")
                f.write(str(summary_stats))
            
            print(f"Saved summary statistics to: {summary_path}")
            
            cultural_coordinates_path = self.results_path / "cultural_coordinates.json"
            country_scores_pca.to_json(cultural_coordinates_path, orient='records', indent=2)
            print(f"Saved cultural coordinates to: {cultural_coordinates_path}")
            
            return True
            
        except Exception as e:
            print(f"Visualization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_complete_analysis(self):
        """Run the full benchmark pipeline."""
        print("Starting country-benchmark pipeline...")
        print(f"Start time: {datetime.now()}")
        
        success_steps = []
        
        if self.step1_data_processing():
            success_steps.append("data processing")
        else:
            print("Stopping after data-processing failure.")
            return False
        
        if self.step2_pca_analysis():
            success_steps.append("PCA analysis")
        else:
            print("Stopping after PCA failure.")
            return False
        
        if self.step3_visualization():
            success_steps.append("visualization")
        else:
            print("Visualization failed, but earlier steps completed successfully.")
        
        print("\n" + "="*60)
        print("Country-benchmark pipeline finished.")
        print("="*60)
        print(f"Completed steps: {', '.join(success_steps)}")
        print(f"Data directory: {self.data_path}")
        print(f"Results directory: {self.results_path}")
        print(f"Finish time: {datetime.now()}")
        
        print("\nGenerated files:")
        
        data_files = [
            "ivs_df.pkl", "variable_view.pkl", "valid_data.pkl", 
            "country_codes.pkl", "country_scores_pca.pkl", "country_scores_pca.json"
        ]
        print("\nData files (data/country_values/):")
        for file_name in data_files:
            file_path = self.data_path / file_name
            if file_path.exists():
                print(f"  present: {file_name}")
            else:
                print(f"  missing: {file_name}")
        
        result_files = [
            "cultural_map.png", "decision_boundary.png", 
            "summary_statistics.txt", "cultural_coordinates.json"
        ]
        print("\nResult files (results/country_values/):")
        for file_name in result_files:
            file_path = self.results_path / file_name
            if file_path.exists():
                print(f"  present: {file_name}")
            else:
                print(f"  missing: {file_name}")
        
        return len(success_steps) == 3


def main():
    """Entry point."""
    
    print("Country Values Analysis Runner")
    print("=" * 60)
    
    try:
        runner = CountryValuesAnalysisRunner()
        
        success = runner.run_complete_analysis()
        
        if success:
            print("\nAll analysis stages completed successfully.")
            return 0
        else:
            print("\nOne or more analysis stages failed. See logs above.")
            return 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\nUnexpected error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
