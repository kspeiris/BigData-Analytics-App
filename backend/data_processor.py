import pandas as pd
import numpy as np
from datetime import datetime

class DataProcessor:
    def __init__(self, data):
        self.data = data
        self.df = pd.DataFrame(data)
    
    def get_basic_info(self):
        return {
            "shape": self.df.shape,
            "columns": self.df.columns.tolist(),
            "data_types": self.df.dtypes.astype(str).to_dict(),
            "missing_values": self.df.isnull().sum().to_dict()
        }
    
    def get_numeric_columns(self):
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_columns(self):
        return self.df.select_dtypes(include=['object']).columns.tolist()
    
    def calculate_statistics(self):
        numeric_cols = self.get_numeric_columns()
        stats = {}
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(self.df[col].mean()),
                'median': float(self.df[col].median()),
                'std': float(self.df[col].std()),
                'min': float(self.df[col].min()),
                'max': float(self.df[col].max()),
                'count': int(self.df[col].count())
            }
        
        return stats