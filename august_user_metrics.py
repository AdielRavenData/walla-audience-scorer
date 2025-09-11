#!/usr/bin/env python3
"""
August User Metrics Generator

This script processes the august_features table to create user behavioral metrics
and outputs the specified columns for further processing.

Usage:
    python august_user_metrics.py

Output columns:
    - user_unique_id
    - region
    - city
    - device_category
    - sector
    - activity_count
    - session_count
    - avg_session_duration
    - weekend_ratio
    - vertical_diversity
    
Note: count_vertical_x, count_category_x, hour_x, day_x, and time_of_day_x columns have been removed per user request.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from google.cloud import bigquery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AugustUserMetricsGenerator:
    """
    Generates user behavioral metrics from august_features table.
    """
    
    def __init__(self, project_id: str = "wallabi-169712"):
        """
        Initialize the metrics generator.
        
        Args:
            project_id (str): Google Cloud Project ID
        """
        self.project_id = project_id
        self.bq_client = bigquery.Client(project=project_id)
        self.src_table = "august_feature_small"
        
        logger.info(f"Initialized AugustUserMetricsGenerator for project: {project_id}")
    
    def fetch_august_data(self, max_users: int = None) -> pd.DataFrame:
        """
        Fetch data from august_features table.
        
        Args:
            max_users (int): Optional limit on number of users to fetch
            
        Returns:
            pd.DataFrame: Raw interaction data
        """
        logger.info(f"Fetching data from {self.src_table}...")
        
        # Build query with optional user limit
        if max_users:
            query = f"""
            SELECT 
                user_unique_id,
                event_date,
                time_stamp,
                vertical_name,
                CategoryName,
                page_title,
                item_title,
                hour_of_day,
                day_of_week,
                region,
                city,
                device_category,
                sector
            FROM `{self.project_id}.UsersClustering.{self.src_table}`
            WHERE user_unique_id IN (
                SELECT DISTINCT user_unique_id 
                FROM `{self.project_id}.UsersClustering.{self.src_table}`
                LIMIT {max_users}
            )
            ORDER BY user_unique_id, time_stamp
            """
        else:
            query = f"""
            SELECT 
                user_unique_id,
                event_date,
                time_stamp,
                vertical_name,
                CategoryName,
                page_title,
                item_title,
                hour_of_day,
                day_of_week,
                region,
                city,
                device_category,
                sector
            FROM `{self.project_id}.UsersClustering.{self.src_table}`
            ORDER BY user_unique_id, time_stamp
            """
        
        try:
            print("⏳ Executing BigQuery query... (this may take several minutes)")
            df = self.bq_client.query(query).to_dataframe()
            logger.info(f"Fetched {len(df):,} interactions for {df['user_unique_id'].nunique():,} users")
            print(f"✅ BigQuery query completed! Fetched {len(df):,} interactions for {df['user_unique_id'].nunique():,} users")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def create_user_vectors(self, df: pd.DataFrame, use_vertical=False, use_category=False) -> pd.DataFrame:
        """
        Creates user feature vectors with behavioral metrics from interaction data.
        Based on the advanced logic from the clustering script.
        
        Args:
            df (pd.DataFrame): Raw interaction data
            use_vertical (bool): Whether to use vertical_name for diversity (disabled by default)
            use_category (bool): Whether to use CategoryName for diversity (disabled by default)
            
        Returns:
            pd.DataFrame: User behavioral metrics with all features
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return pd.DataFrame()

        if 'user_unique_id' not in df.columns:
            logger.error("user_unique_id column is required but not found")
            raise ValueError("user_unique_id column is required")

        logger.info(f"Processing {len(df)} rows for {df['user_unique_id'].nunique()} unique users")

        # Filter out unwanted data
        if 'vertical_name' in df.columns:
            df = df[df['vertical_name'] != 'וואלה']
            df = df[~df['vertical_name'].str.contains('חדשות', na=False)]



        if df.empty:
            logger.warning("DataFrame is empty after filtering")
            return pd.DataFrame()

        # Convert time_stamp to datetime if needed
        if 'time_stamp' in df.columns:
            try:
                # Check if time_stamp is numeric (unix timestamp)
                if pd.api.types.is_numeric_dtype(df['time_stamp']):
                    # Check if it's seconds, milliseconds, or microseconds
                    sample_value = df['time_stamp'].iloc[0]
                    if sample_value > 1e15:  # microseconds
                        df['datetime'] = pd.to_datetime(df['time_stamp'] / 1000000, unit='s')
                    elif sample_value > 1e12:  # milliseconds
                        df['datetime'] = pd.to_datetime(df['time_stamp'] / 1000, unit='s')
                    else:  # seconds
                        df['datetime'] = pd.to_datetime(df['time_stamp'], unit='s')
                else:
                    # Try to parse as datetime string
                    df['datetime'] = pd.to_datetime(df['time_stamp'])
                logger.info("Successfully converted time_stamp to datetime")
            except Exception as e:
                logger.warning(f"Could not convert time_stamp: {e}")
                df['datetime'] = None
        else:
            logger.warning("time_stamp column not found, using event_date as fallback")
            if 'event_date' in df.columns:
                try:
                    df['datetime'] = pd.to_datetime(df['event_date'])
                except Exception as e:
                    logger.warning(f"Could not convert event_date: {e}")
                    df['datetime'] = None

        # Create time-based features from datetime
        if 'datetime' in df.columns and df['datetime'] is not None:
            try:
                if 'day_of_week' not in df.columns:
                    df['day_of_week'] = df['datetime'].dt.dayofweek
                if 'hour_of_day' not in df.columns:
                    df['hour_of_day'] = df['datetime'].dt.hour
                logger.info("Created day_of_week and hour_of_day from datetime")
            except Exception as e:
                logger.warning(f"Could not create time-based columns: {e}")

        # Sort for session calculations using datetime
        if 'datetime' in df.columns and df['datetime'] is not None:
            df = df.sort_values(['user_unique_id', 'datetime'])
            logger.info("Sorted data by user_unique_id and datetime for session calculations")

        logger.info("Calculating behavioral metrics...")
        feature_dfs = []

        # 2. Vertical Count Features - REMOVED per user request
        # 3. Category Count Features - REMOVED per user request

        # 4. Time-based Features
        df_sorted = df.sort_values(['user_unique_id', 'datetime']) if 'datetime' in df.columns else df.copy()
        
        # 1. Basic Activity Count
        activity_counts = df_sorted.groupby('user_unique_id').size().rename('activity_count')
        feature_dfs.append(activity_counts.to_frame())
        logger.info("Created activity count features")
        
        # Create derived time columns if possible
        if 'datetime' in df.columns and 'day_of_week' not in df.columns:
            df_sorted['day_of_week'] = df_sorted['datetime'].dt.dayofweek
        
        if 'datetime' in df.columns and 'hour_of_day' not in df.columns:
            df_sorted['hour_of_day'] = df_sorted['datetime'].dt.hour
        
        # Only calculate time features if we have the necessary date column
        if 'datetime' in df.columns:
            df_sorted['next_timestamp'] = df_sorted.groupby('user_unique_id')['datetime'].shift(-1)
            df_sorted['time_diff'] = (df_sorted['next_timestamp'] - df_sorted['datetime']).dt.total_seconds()
            df_sorted['time_diff'] = df_sorted['time_diff'].clip(upper=1800)  # 30 minutes max
            logger.info("Calculated time differences between consecutive events")

            # Time-based vertical features - REMOVED per user request
        
        # 5. Day of Week Features - REMOVED per user request
        
        # 6. Hour of Day Features - REMOVED per user request

        # 7. Advanced Behavioral Features
        if 'datetime' in df.columns and df['datetime'] is not None:
            # Calculate session-based features
            if 'time_diff' in df_sorted.columns:
                df_sorted['session_start'] = df_sorted['time_diff'].isna() | (df_sorted['time_diff'] > 1800)
                df_sorted['session_id'] = df_sorted.groupby('user_unique_id')['session_start'].cumsum()
                logger.info("Created session IDs based on 30-minute gaps")

                # Session counts per user
                user_session_counts = df_sorted.groupby('user_unique_id')['session_id'].nunique().to_frame('session_count')
                feature_dfs.append(user_session_counts)
                logger.info("Calculated session counts per user")

                # Average session duration
                session_durations = df_sorted.groupby(['user_unique_id', 'session_id'], observed=True)['time_diff'].sum()
                avg_session_duration = session_durations.groupby('user_unique_id').mean().to_frame('avg_session_duration')
                feature_dfs.append(avg_session_duration)
                logger.info("Calculated average session durations")

            # Time of day preferences - REMOVED per user request

            # Calculate weekday vs weekend ratio if day_of_week is available
            if 'day_of_week' in df_sorted.columns:
                df_sorted['is_weekend'] = df_sorted['day_of_week'].isin([5, 6])  # Assuming 5=Friday, 6=Saturday
                weekday_counts = df_sorted.groupby(['user_unique_id', 'is_weekend'], observed=True).size().unstack(fill_value=0)
                
                # Calculate weekend ratio
                weekend_ratio = pd.Series(index=weekday_counts.index, dtype=float)
                
                if 1 in weekday_counts.columns and 0 in weekday_counts.columns:
                    # Both weekday and weekend data available
                    weekend_ratio = weekday_counts[1] / (weekday_counts[0] + 1)
                elif 1 in weekday_counts.columns:
                    # Only weekend data available
                    weekend_ratio[:] = 1.0
                elif 0 in weekday_counts.columns:
                    # Only weekday data available
                    weekend_ratio[:] = 0.0
                else:
                    # No data (shouldn't happen)
                    weekend_ratio[:] = 0.0
                
                feature_dfs.append(weekend_ratio.to_frame('weekend_ratio'))
                logger.info("Created weekend ratio features")

            # Vertical diversity - how many different verticals does the user visit
            if 'vertical_name' in df.columns:
                vertical_diversity = df.groupby('user_unique_id')['vertical_name'].nunique().to_frame('vertical_diversity')
                feature_dfs.append(vertical_diversity)
                logger.info("Created vertical diversity features")

        # 8. User Demographics (most frequent value per user)
        demographic_cols = ['region', 'city', 'device_category', 'sector']
        
        for col in demographic_cols:
            if col in df.columns:
                try:
                    # Get most frequent value per user
                    most_frequent = df.groupby('user_unique_id')[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
                    feature_dfs.append(most_frequent.to_frame(col))
                except Exception as e:
                    logger.warning(f"Could not calculate most frequent {col}: {e}")
                    feature_dfs.append(pd.Series('Unknown', index=activity_counts.index, name=col).to_frame())
            else:
                feature_dfs.append(pd.Series('Unknown', index=activity_counts.index, name=col).to_frame())
        
        # Combine all features
        if feature_dfs:
            user_features = pd.concat([df for df in feature_dfs if not df.empty], axis=1)
            user_features = user_features.fillna(0)
        else:
            # If no features were created, create a basic DataFrame with user_unique_id
            user_ids = df['user_unique_id'].unique()
            user_features = pd.DataFrame(index=user_ids)
            # Add at least one feature column to avoid empty DataFrame issues
            user_features['activity_count'] = df.groupby('user_unique_id').size()
        
        # Fill missing values and ensure reasonable ranges
        user_features = user_features.fillna({
            'activity_count': 1,
            'session_count': 1,
            'avg_session_duration': 300.0,  # Default 5-minute sessions
            'weekend_ratio': 0.3,
            'vertical_diversity': 1,
            'region': 'Unknown',
            'city': 'Unknown',
            'device_category': 'Unknown',
            'sector': 'Unknown'
        })

        # Ensure numeric columns have reasonable ranges
        if 'session_count' in user_features.columns:
            user_features['session_count'] = user_features['session_count'].clip(lower=1)
        if 'avg_session_duration' in user_features.columns:
            user_features['avg_session_duration'] = user_features['avg_session_duration'].clip(lower=300, upper=7200)  # Default 5-minute sessions
        if 'weekend_ratio' in user_features.columns:
            user_features['weekend_ratio'] = user_features['weekend_ratio'].clip(lower=0, upper=1)
        if 'vertical_diversity' in user_features.columns:
            user_features['vertical_diversity'] = user_features['vertical_diversity'].clip(lower=1)

        logger.info(f"Created user feature vectors with {user_features.shape[1]} features for {user_features.shape[0]} users")
        return user_features
    
    def generate_metrics(self, max_users: int = None, output_file: str = "august_user_metrics.csv") -> pd.DataFrame:
        """
        Main method to generate user metrics from august data.
        
        Args:
            max_users (int): Optional limit on number of users
            output_file (str): Output CSV file path
            
        Returns:
            pd.DataFrame: User behavioral metrics
        """
        logger.info("Starting August user metrics generation...")
        print("🚀 Starting August user metrics generation...")
        
        try:
            # Step 1: Fetch data
            print("📊 Step 1/3: Fetching data from august_feature...")
            df = self.fetch_august_data(max_users)
            
            if df.empty:
                logger.warning("No data found")
                print("⚠️ No data found")
                return pd.DataFrame()
            
            # Step 2: Create user vectors
            print("🔄 Step 2/3: Creating user behavioral metrics...")
            user_metrics = self.create_user_vectors(df)
            
            if user_metrics.empty:
                logger.warning("No user metrics generated")
                print("⚠️ No user metrics generated")
                return pd.DataFrame()
            
            # Step 3: Save results
            print("💾 Step 3/3: Saving results...")
            user_metrics.to_csv(output_file, index=True)  # index=True to include user_unique_id
            print(f"✅ Results saved to: {output_file}")
            
            # Show summary
            print(f"\n📈 Summary:")
            print(f"Total users: {len(user_metrics):,}")
            print(f"Total features: {user_metrics.shape[1]}")
            print(f"Average activity count: {user_metrics['activity_count'].mean():.1f}")
            
            # Show available metrics
            if 'session_count' in user_metrics.columns:
                print(f"Average session count: {user_metrics['session_count'].mean():.1f}")
            if 'avg_session_duration' in user_metrics.columns:
                print(f"Average session duration: {user_metrics['avg_session_duration'].mean():.1f} seconds")
            if 'weekend_ratio' in user_metrics.columns:
                print(f"Average weekend ratio: {user_metrics['weekend_ratio'].mean():.3f}")
            if 'vertical_diversity' in user_metrics.columns:
                print(f"Average vertical diversity: {user_metrics['vertical_diversity'].mean():.1f}")
            
            # Show feature breakdown
            feature_types = {}
            for col in user_metrics.columns:
                if col in ['region', 'city', 'device_category', 'sector']:
                    feature_types['Demographics'] = feature_types.get('Demographics', 0) + 1
                else:
                    feature_types['Other'] = feature_types.get('Other', 0) + 1
            
            print(f"\n📊 Feature Breakdown:")
            for feature_type, count in feature_types.items():
                print(f"  {feature_type}: {count} features")
            
            logger.info(f"August user metrics generation complete! Generated metrics for {len(user_metrics)} users")
            return user_metrics
            
        except Exception as e:
            logger.error(f"Error in metrics generation: {e}")
            raise


def main():
    """Interactive command-line interface."""
    print("=== August User Metrics Generator ===")
    print("Generate user behavioral metrics from august_feature table")
    print()
    
    # Get user inputs
    try:
        max_users = input("Enter max users to process (default: all users): ").strip()
        max_users = int(max_users) if max_users else None
    except ValueError:
        max_users = None
    
    output_file = input("Enter output CSV file (default: august_user_metrics.csv): ").strip() or "august_user_metrics.csv"
    
    print(f"\nStarting metrics generation...")
    print(f"Max users: {max_users if max_users else 'All users'}")
    print(f"Output file: {output_file}")
    print("-" * 50)
    
    # Initialize generator and run analysis
    generator = AugustUserMetricsGenerator()
    results = generator.generate_metrics(max_users, output_file)
    
    if results.empty:
        print("No results generated. Please check your inputs and try again.")
        return
    
    # Show sample of results
    print(f"\n📋 Sample Results (first 5 users):")
    # Only show columns that exist in the results
    available_cols = ['user_unique_id', 'region', 'city', 'device_category', 'sector', 'activity_count']
    optional_cols = ['session_count', 'avg_session_duration', 'weekend_ratio', 'vertical_diversity']
    
    # Add optional columns if they exist
    for col in optional_cols:
        if col in results.columns:
            available_cols.append(col)
    
    sample_results = results.head().reset_index()
    print(sample_results[available_cols].to_string(index=False))
    
    return results


if __name__ == "__main__":
    results = main()
