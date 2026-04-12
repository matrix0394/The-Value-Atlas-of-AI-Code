import os
import pandas as pd
import pyreadstat
import pickle
import json

class DataProcessor:
    """Load IVS data and prepare the country-level benchmark inputs."""

    def __init__(self, data_path="data/country_values"):
        self.project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if not os.path.isabs(data_path):
            self.data_path = os.path.join(self.project_root, data_path)
        else:
            self.data_path = data_path

        os.makedirs(self.data_path, exist_ok=True)
        print(f"Data path resolved to: {self.data_path}")

        self.ivs_df = None
        self.variable_view = None
        self.filtered_data = None

    def load_ivs_data(self, ivs_file="Integrated_values_surveys_1981-2022.sav"):
        """Load the IVS `.sav` file and build a variable summary table."""
        ivs_path = os.path.join(self.data_path, ivs_file)
        print(f"Loading IVS data from {ivs_path}...")

        ivs_data, ivs_meta = pyreadstat.read_sav(ivs_path, encoding='latin1')
        self.ivs_df = pd.DataFrame(ivs_data)

        self._create_variable_view(ivs_meta)

        print(f"Loaded {len(self.ivs_df)} observations with {len(self.ivs_df.columns)} variables")
        return self.ivs_df

    def _create_variable_view(self, ivs_meta):
        """Create a compact variable-view table similar to SPSS metadata output."""
        variable_names = ivs_meta.column_names
        variable_labels = ivs_meta.column_labels
        variable_types = [ivs_meta.readstat_variable_types[var] for var in variable_names]
        variable_measure = [ivs_meta.variable_measure.get(var, 'None') for var in variable_names]
        variable_alignment = [ivs_meta.variable_alignment.get(var, 'None') for var in variable_names]
        variable_display_width = [ivs_meta.variable_display_width.get(var, 'None') for var in variable_names]
        missing_values = [ivs_meta.missing_user_values.get(var, 'None') for var in variable_names]

        variable_types = ['Numeric' if vtype == 'double' else vtype.capitalize() for vtype in variable_types]

        self.variable_view = pd.DataFrame({
            'Name': variable_names,
            'Type': variable_types,
            'Width': variable_display_width,
            'Label': variable_labels,
            'Missing': missing_values,
            'Measure': variable_measure,
            'Align': variable_alignment
        })

    def save_data(self):
        """Persist the loaded and filtered tables used by the benchmark pipeline."""
        if self.ivs_df is not None:
            ivs_path = os.path.join(self.data_path, "ivs_df.pkl")
            self.ivs_df.to_pickle(ivs_path)
            print(f"Saved ivs_df to {ivs_path}")
        
        if self.variable_view is not None:
            var_path = os.path.join(self.data_path, "variable_view.pkl")
            self.variable_view.to_pickle(var_path)
            print(f"Saved variable_view to {var_path}")

        if self.filtered_data is not None:
            valid_data_path = os.path.join(self.data_path, "valid_data.pkl")
            self.filtered_data.to_pickle(valid_data_path)
            print(f"Saved valid_data to {valid_data_path}")

    def get_filtered_data(self, year_threshold=2005):
        """Filter the IVS table to the subset used for PCA estimation."""
        if self.ivs_df is None:
            raise ValueError("Please load data first")

        meta_col = ["S020", "S003"]
        weights = ["S017"]
        iv_qns = ["A008", "A165", "E018", "E025", "F063", "F118", "F120", "G006", "Y002", "Y003"]

        subset_ivs_df = self.ivs_df[meta_col + weights + iv_qns].copy()
        subset_ivs_df = subset_ivs_df.rename(columns={'S020': 'year', 'S003': 'country_code', 'S017': 'weight'})

        subset_ivs_df = subset_ivs_df[subset_ivs_df["year"] >= year_threshold]

        subset_ivs_df = subset_ivs_df.dropna(subset=iv_qns, thresh=6)

        print(f"Filtered data: {len(subset_ivs_df)} observations from {len(subset_ivs_df['country_code'].unique())} countries")

        self.filtered_data = subset_ivs_df

        return subset_ivs_df

    def create_country_codes(self):
        """Build the country-code table used across the project configuration."""
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, 'config', 'country', 'cultural_regions.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        numeric_to_region = {}
        for region, codes in config['cultural_regions'].items():
            for code in codes:
                numeric_to_region[code] = region

        islamic_countries_list = config.get('islamic_countries', [])
        islamic_countries = {country: True for country in islamic_countries_list}

        country_codes_mapping = config.get('country_codes', {})

        country_codes = pd.DataFrame({
            'Country': list(country_codes_mapping.keys()),
            'Numeric': list(country_codes_mapping.values())
        })

        country_codes['Cultural Region'] = country_codes['Numeric'].map(numeric_to_region).fillna('Other')

        country_codes['Islamic'] = country_codes['Country'].map(islamic_countries).fillna(False)

        config_dir = os.path.join(project_root, 'config', 'country')
        country_codes_path = os.path.join(config_dir, "country_codes.pkl")
        country_codes.to_pickle(country_codes_path)
        print(f"Country codes saved to {country_codes_path} with {len(country_codes)} countries")

        country_codes_json_path = os.path.join(config_dir, "country_codes.json")
        country_codes.to_json(country_codes_json_path, orient='records', indent=2, force_ascii=False)
        print(f"Country codes JSON saved to {country_codes_json_path}")

        return country_codes

def main():
    """Run a small end-to-end smoke test for IVS preprocessing."""
    print("=== Testing IVS data processing ===")

    processor = DataProcessor()

    try:
        print("\n1. Loading IVS data...")
        ivs_df = processor.load_ivs_data()
        print(f"Loaded data: {len(ivs_df)} rows, {len(ivs_df.columns)} columns")

        print("\n2. Checking country-code metadata...")
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_country_codes_path = os.path.join(project_root, 'config', 'country', 'country_codes.pkl')
        if not os.path.exists(config_country_codes_path):
            country_codes = processor.create_country_codes()
            print(f"Created country_codes with {len(country_codes)} countries")
        else:
            print("country_codes already exists in config/country")

        print("\n3. Filtering the IVS sample...")
        filtered_data = processor.get_filtered_data()
        print(f"Filtered data: {len(filtered_data)} rows")

        print("\n4. Saving intermediate files...")
        processor.save_data()
        print("Saved processed outputs")

        print("\n5. Summary:")
        print(f"Original data shape: {ivs_df.shape}")
        print(f"Filtered data shape: {filtered_data.shape}")
        print(f"Countries included: {sorted(filtered_data['country_code'].unique())}")

    except Exception as e:
        print(f"Data-processing test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
