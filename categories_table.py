#!/usr/bin/env python3
"""
Categories Audience Summary Generator

This script creates a summary table of category counts per user from the users_with_audience_score table
and appends the results to a categories_audience_summary table in BigQuery.

Usage:
    python categories_table.py
"""

import pandas as pd
import logging
from google.cloud import bigquery
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CategoriesAudienceSummaryGenerator:
    """
    Generator for creating categories audience summary from BigQuery data.
    """
    
    def __init__(self, project_id: str = "wallabi-169712"):
        """
        Initialize the generator.
        
        Args:
            project_id (str): Google Cloud project ID
        """
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        logger.info(f"Initialized CategoriesAudienceSummaryGenerator for project: {project_id}")
    
    def generate_categories_summary(self, run_id: int, audience: str) -> pd.DataFrame:
        """
        Generate categories audience summary from BigQuery data.
        
        Args:
            run_id (int): Run ID to use
            audience (str): Audience name to use
            
        Returns:
            pd.DataFrame: DataFrame with categories summary data
        """
        try:
            logger.info(f"Generating categories audience summary with run_id: {run_id}, audience: {audience}")
            
            # SQL query to get category counts per user
            query = f"""
            WITH run_users AS (
                SELECT DISTINCT user_unique_id 
                FROM `{self.project_id}.UsersClustering.users_with_audience_score`
                WHERE audience = '{audience}' AND run_id = {run_id}
            )
            SELECT  
                a.user_unique_id,
                CategoryName as category_name,
                {run_id} as run_id,
                '{audience}' as audience,
                COUNT(*) as category_count
            FROM `{self.project_id}.UsersClustering.august_feature_first_week` a
            JOIN run_users ru ON ru.user_unique_id = a.user_unique_id
            GROUP BY 1, 2, 3, 4
            ORDER BY a.user_unique_id, category_count DESC
            """
            
            logger.info("Executing BigQuery query...")
            print("⏳ Executing BigQuery query... (this may take several minutes)")
            
            # Execute query
            df = self.bq_client.query(query).to_dataframe()
            
            logger.info(f"Query completed! Fetched {len(df):,} category summary records")
            print(f"✅ BigQuery query completed! Fetched {len(df):,} category summary records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error generating categories summary: {e}")
            raise
    
    def append_to_bigquery_table(self, df: pd.DataFrame, table_name: str = "categories_audience_summary") -> None:
        """
        Append results to BigQuery table.
        
        Args:
            df: DataFrame with categories summary data
            table_name: BigQuery table name
        """
        try:
            # Prepare table reference
            table_id = f"{self.project_id}.UsersClustering.{table_name}"
            
            # Clean data types for BigQuery compatibility
            print("🧹 Cleaning data types for BigQuery compatibility...")
            df_to_upload = df.copy()
            
            for col in df_to_upload.columns:
                if df_to_upload[col].dtype == 'object':
                    # Convert object columns to string and handle NaN values
                    df_to_upload[col] = df_to_upload[col].astype(str).replace('nan', 'Unknown')
                elif 'datetime' in str(df_to_upload[col].dtype):
                    # Convert datetime columns to string
                    df_to_upload[col] = df_to_upload[col].astype(str)
            
            # Upload data
            print(f"📊 Uploading {len(df_to_upload.columns)} columns: {list(df_to_upload.columns)}")
            print(f"📤 Uploading {len(df_to_upload):,} rows to {table_id}...")
            
            # Configure job
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",  # Append mode
                create_disposition="CREATE_IF_NEEDED",  # Create table if doesn't exist
                autodetect=True  # Auto-detect schema
            )
            
            # Upload data
            job = self.bq_client.load_table_from_dataframe(
                df_to_upload, 
                table_id, 
                job_config=job_config
            )
            
            # Wait for job to complete
            job.result()
            
            logger.info(f"Successfully uploaded {len(df_to_upload):,} rows to {table_id}")
            print(f"✅ Successfully uploaded {len(df_to_upload):,} rows to {table_id}")
            
        except Exception as e:
            logger.error(f"Error appending to BigQuery table: {e}")
            print(f"❌ Error appending to BigQuery table: {e}")
            raise
    
    def run(self, run_id: int, audience: str, table_name: str = "categories_audience_summary") -> pd.DataFrame:
        """
        Main method to generate and upload categories audience summary.
        
        Args:
            run_id (int): Run ID to use
            audience (str): Audience name to use
            table_name (str): BigQuery table name for appending results
            
        Returns:
            pd.DataFrame: Generated summary data
        """
        try:
            print("🚀 Starting Categories Audience Summary generation...")
            
            # Step 1: Generate summary data
            print("📊 Step 1/2: Generating categories summary from BigQuery...")
            summary_df = self.generate_categories_summary(run_id, audience)
            
            if summary_df.empty:
                print("⚠️ No summary data generated")
                return pd.DataFrame()
            
            # Step 2: Upload to BigQuery
            print("📤 Step 2/2: Appending to BigQuery table...")
            self.append_to_bigquery_table(summary_df, table_name)
            
            # Summary
            print(f"\n📈 Summary:")
            print(f"Run ID: {run_id}")
            print(f"Audience: {audience}")
            print(f"Total records: {len(summary_df):,}")
            print(f"Unique users: {summary_df['user_unique_id'].nunique():,}")
            print(f"Unique categories: {summary_df['category_name'].nunique():,}")
            print(f"Average categories per user: {len(summary_df) / summary_df['user_unique_id'].nunique():.1f}")
            
            logger.info("Categories audience summary generation complete!")
            return summary_df
            
        except Exception as e:
            logger.error(f"Error in categories summary generation: {e}")
            raise

def main():
    """Main function to run the categories audience summary generator."""
    try:
        # Create generator
        generator = CategoriesAudienceSummaryGenerator()
        
        # Ask user for inputs
        print("🚀 Starting Categories Audience Summary generation...")
        
        # Get run_id from user
        run_id_input = input("Enter run_id (or press Enter for '1'): ").strip()
        run_id = int(run_id_input) if run_id_input else 1
        
        # Get audience from user with predefined options
        print("\nAvailable audience options:")
        print("1. אקדמאים (Academics)")
        print("2. הייטקיסטים (Tech Professionals)")
        
        audience_choice = input("Select audience (1 or 2, or press Enter for 'אקדמאים'): ").strip()
        
        if audience_choice == "2":
            audience = "הייטקיסטים"
        else:
            audience = "אקדמאים"  # Default option
        
        # Ask user for table name
        table_name = "categories_audience_summary"
        custom_table = input(f"Enter custom table name (or press Enter for '{table_name}'): ").strip()
        table_name = custom_table if custom_table else table_name
        
        print(f"\nStarting categories summary generation...")
        print(f"Run ID: {run_id}")
        print(f"Audience: {audience}")
        print(f"Table name: {table_name}")
        print("--------------------------------------------------")
        
        # Run the generator
        results = generator.run(run_id, audience, table_name)
        
        if not results.empty:
            print(f"\n🎉 Success! Generated {len(results):,} category summary records")
        else:
            print("\n⚠️ No results generated")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
