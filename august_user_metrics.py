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
    
    # Academic-relevant verticals for filtering (expanded from 3 to 16)
    ACADEMIC_VERTICALS = [
        # Cultural & Arts
        'תרבות', 'יהדות', 'פיס בתרבות',
        # Professional & Business  
        'קריירה', 'המומחים', 'זירת היועצים', 'עסקים קטנים', 'marketing',
        # Health & Science
        'בריאות', 'מדעני העתיד',
        # Other Academic
        'הנבחרים', 'טוב לדעת', 'עזרה', 'אולימפיאדת טוקיו 2020', 'מגזין', 'משפט'
    ]
    
    # Academic keywords for content scoring
    STEM_KEYWORDS = [
        # Core academic terms
        'אקדמ', 'אוניברסיט', 'מחקר', 'מדע', 'מדעי', 'פילוסופ', 'היסטור', 'סוציולוג', 'כלכל', 'ביולוג', 'כימ', 'פיז',
        # Education and learning
        'חינוך', 'לימוד', 'קורס', 'הרצאה', 'כנס', 'סמינר', 'השכלה', 'תואר', 'דוקטור', 'פרופסור',
        # Culture and arts
        'תרבות', 'אמנות', 'מוזיקה', 'שירה', 'ספר', 'ספרות', 'תיאטרון', 'מחול', 'בלט', 'אופרה',
        # Analysis and criticism
        'ביקורת', 'ניתוח', 'מאמר', 'כתב', 'מגזין', 'כתב עת', 'סקירה', 'דעה', 'דיון', 'ויכוח',
        # Research and study
        'סקר', 'סטטיסטיקה', 'נתונים', 'ממצאים', 'תוצאות', 'מסקנות', 'המלצות', 'הערכה'
    ]
    
    EXACT_KEYWORDS = [
        # Academic institutions and concepts
        'אוניברסיטה', 'מכללה', 'מכון', 'מרכז מחקר', 'מעבדה', 'ספרייה', 'ארכיון', 'מוזיאון',
        # Cultural and artistic terms
        'ביקורת ספרים', 'ביקורת הופעה', 'ביקורת סרט', 'ביקורת תיאטרון', 'ביקורת אמנות',
        'תרבות ישראלית', 'תרבות עברית', 'אמנות ישראלית', 'מוזיקה ישראלית', 'שירה עברית',
        # Academic writing
        'מאמר אקדמי', 'מחקר אקדמי', 'דוקטורט', 'תזה', 'דיסרטציה', 'מונוגרפיה',
        # Historical and philosophical terms
        'היסטוריה ישראלית', 'היסטוריה יהודית', 'פילוסופיה יהודית', 'מחשבת ישראל',
        # Literary terms
        'ספרות עברית', 'ספרות ישראלית', 'שירה עברית', 'פרוזה', 'שירה', 'רומן', 'נובלה',
        # Academic events
        'כנס אקדמי', 'הרצאה אקדמית', 'סמינר אקדמי', 'קונגרס', 'סימפוזיון'
    ]
    
    # Negative keywords for academic content filtering
    NEGATIVE_KEYWORDS = [
        # Car and transportation (very common in dataset)
        'רכב', 'מכונית', 'טיגו', 'היילקס', 'היברידי', 'חשמלית', 'דרכים', 'מבחן', 'מבחן דרכים', 'חוות דעת',
        # Sports
        'כדורגל', 'כדורסל', 'ליגה', 'שער', 'משחק', 'ספורט', 'מכבי', 'הפועל',
        # Food and cooking
        'מתכון', 'מסעד', 'בישול', 'השווארמ', 'אוכל', 'מסעדה', 'בישול',
        # Entertainment and celebrities
        'צפייה ישירה', 'VOD', 'פרק', 'סדרה', 'טריילר', 'לייב', 'פלייבוי', 'סלב', 'סלבס',
        # Shopping and commerce
        'שופינג', 'מבצע', 'חינם', 'קניון', 'קנייה', 'מחיר', 'שקל', 'אלף שקל',
        # Technology (unless academic)
        'iPhone', 'אנדרואיד', 'סמארטפון', 'אפליקציה', 'גיימינג', 'משחקים'
    ]
    
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
    
    def fetch_august_data(self, max_users: int = None, academic_verticals_only: bool = False) -> pd.DataFrame:
        """
        Fetch data from august_features table.
        
        Args:
            max_users (int): Optional limit on number of users to fetch
            academic_verticals_only (bool): If True, only fetch users with academic vertical interactions
            
        Returns:
            pd.DataFrame: Raw interaction data
        """
        logger.info(f"Fetching data from {self.src_table}...")
        
        # Build query with optional user limit and academic verticals filtering
        if academic_verticals_only:
            verticals_list = "', '".join(self.ACADEMIC_VERTICALS)
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
                    WHERE vertical_name IN ('{verticals_list}')
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
                WHERE user_unique_id IN (
                    SELECT DISTINCT user_unique_id 
                    FROM `{self.project_id}.UsersClustering.{self.src_table}`
                    WHERE vertical_name IN ('{verticals_list}')
                )
                ORDER BY user_unique_id, time_stamp
                """
        else:
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
    
    def calculate_audience_scores(self, df: pd.DataFrame, audience: str = 'אקדמאים', min_distinct_categories: int = 1, filter_verticals: bool = True) -> pd.DataFrame:
        """
        Calculate academic audience scores based on page title keyword matching.
        
        Args:
            df: DataFrame with user interaction data
            audience: Target audience name (default: 'אקדמאים' - academic)
            min_distinct_categories: Minimum distinct categories required (default: 1)
            filter_verticals: Whether to filter by academic-relevant verticals (default: True)
            
        Returns:
            pd.DataFrame: User academic audience scores with columns [user_unique_id, total_titles, audience_titles, score]
        """
        import re
        
        # Use class constants for academic keyword patterns
        
        
        # Store original dataframe for total views calculation
        df_original = df.copy()
        
        # Filter by academic-relevant verticals if requested
        if filter_verticals and 'vertical_name' in df.columns:
            original_count = len(df)
            df_filtered = df[df['vertical_name'].isin(self.ACADEMIC_VERTICALS)].copy()
            filtered_count = len(df_filtered)
            logger.info(f"Filtered to academic verticals: {original_count} -> {filtered_count} rows ({filtered_count/original_count*100:.1f}%)")
            
            if filtered_count == 0:
                logger.warning("No data remaining after vertical filtering")
                return pd.DataFrame()
            
            df = df_filtered
        else:
            logger.info("Using all verticals for audience scoring")
        
        # Filter users with enough categories
        if 'CategoryName' not in df.columns or 'page_title' not in df.columns:
            logger.warning("CategoryName or page_title columns not found, skipping audience scoring")
            return pd.DataFrame()
        
        # Clean and filter data
        user_titles = df[
            (df['page_title'].notna()) & 
            (df['page_title'].str.strip() != '') &
            (df['CategoryName'].notna()) & 
            (df['CategoryName'].str.strip() != '')
        ].copy()
        
        if user_titles.empty:
            logger.warning("No valid page titles found for audience scoring")
            return pd.DataFrame()
        
        # Normalize page titles
        user_titles['page_title_norm'] = user_titles['page_title'].str.strip().str.replace(r'["״׳\s]+', ' ', regex=True)
        user_titles['category_name'] = user_titles['CategoryName'].str.strip()
        
        # Filter users with enough distinct categories
        user_category_counts = user_titles.groupby('user_unique_id')['category_name'].nunique()
        eligible_users = user_category_counts[user_category_counts >= min_distinct_categories].index
        
        if len(eligible_users) == 0:
            logger.warning(f"No users found with >= {min_distinct_categories} distinct categories")
            return pd.DataFrame()
        
        # Filter to eligible users only
        user_titles = user_titles[user_titles['user_unique_id'].isin(eligible_users)]
        
        # Create regex patterns
        def escape_regex(text):
            return re.escape(text)
        
        # Exact keywords pattern
        exact_pattern = '|'.join(escape_regex(kw) for kw in self.EXACT_KEYWORDS)
        exact_regex = f'(?i)(^|[^א-ת])({exact_pattern})([^א-ת]|$)'
        
        # Stem keywords pattern  
        stem_pattern = '|'.join(escape_regex(kw) + '[א-ת]*' for kw in self.STEM_KEYWORDS)
        stem_regex = f'(?i)(^|[^א-ת])({stem_pattern})([^א-ת]|$)'
        
        # Negative keywords pattern
        neg_pattern = '|'.join(escape_regex(kw) for kw in self.NEGATIVE_KEYWORDS)
        neg_regex = f'(?i)({neg_pattern})'
        
        # Score each page title
        def score_title(title):
            if pd.isna(title):
                return False
            
            # Check for exact or stem matches
            has_positive = bool(re.search(exact_regex, title) or re.search(stem_regex, title))
            
            # Check for negative matches
            has_negative = bool(re.search(neg_regex, title))
            
            # Your logic: if we have positive keyword, it's academic (ignore negative)
            # Only if we have ONLY negative keywords, then it's not academic
            return has_positive
        
        # Apply scoring
        user_titles['is_audience'] = user_titles['page_title_norm'].apply(score_title)
        
        # Calculate scores per user (only on filtered verticals)
        user_scores = user_titles.groupby('user_unique_id').agg({
            'page_title_norm': 'count',  # This is now the filtered count
            'is_audience': 'sum'        # This is the academic count from filtered data
        }).rename(columns={
            'page_title_norm': 'audience_titles',  # Views in academic-relevant verticals
            'is_audience': 'academic_titles'       # Academic content from filtered verticals
        })
        
        # Get total views from ALL verticals (not just filtered ones)
        if filter_verticals:
            # Get total page views from original dataset (all verticals)
            total_views = df_original.groupby('user_unique_id').size().rename('total_views')
            user_scores = user_scores.join(total_views, how='left')
            user_scores['total_views'] = user_scores['total_views'].fillna(0)
        else:
            # If not filtering, total views = audience titles
            user_scores['total_views'] = user_scores['audience_titles']
        
        # Calculate score as ratio of academic content from filtered verticals
        user_scores['score'] = user_scores['academic_titles'] / user_scores['audience_titles']
        
        # Add audience name
        user_scores['audience'] = audience
        
        # Add score date
        user_scores['score_date'] = pd.Timestamp.now().strftime('%Y-%m-%d')
        
        # Rename columns to match expected output
        user_scores['audience_views'] = user_scores['academic_titles']
        
        logger.info(f"Calculated audience scores for {len(user_scores)} users")
        
        return user_scores.reset_index()


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
        
        # Calculate first and last view dates
        date_stats = df_sorted.groupby('user_unique_id')['event_date'].agg(['min', 'max']).rename(columns={
            'min': 'first_view',
            'max': 'last_view'
        })
        feature_dfs.append(date_stats)
        logger.info("Created date range features")
        
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
                    # Get most frequent value per user (count-based, not just first mode)
                    most_frequent = df.groupby('user_unique_id')[col].agg(lambda x: x.value_counts().index[0] if len(x) > 0 and len(x.value_counts()) > 0 else 'Unknown')
                    feature_dfs.append(most_frequent.to_frame(col))
                except Exception as e:
                    logger.warning(f"Could not calculate most frequent {col}: {e}")
                    feature_dfs.append(pd.Series('Unknown', index=activity_counts.index, name=col).to_frame())
            else:
                feature_dfs.append(pd.Series('Unknown', index=activity_counts.index, name=col).to_frame())
        
        # Combine all features
        if feature_dfs:
            user_features = pd.concat([df for df in feature_dfs if not df.empty], axis=1)
            # Fill missing values (handle date columns separately)
            for col in user_features.columns:
                if 'date' in str(user_features[col].dtype).lower() or col in ['first_view', 'last_view']:
                    # For date columns, don't fill - leave as is
                    continue
                else:
                    # For numeric columns, fill with 0
                    user_features[col] = user_features[col].fillna(0)
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
    
    def generate_metrics(self, max_users: int = None, output_file: str = "august_user_metrics.csv", append_to_table: bool = False, table_name: str = "users_with_audience_score") -> pd.DataFrame:
        """
        Main method to generate user metrics from august data.
        
        Args:
            max_users (int): Optional limit on number of users
            output_file (str): Output CSV file path
            append_to_table (bool): Whether to append results to BigQuery table
            table_name (str): BigQuery table name for appending results
            
        Returns:
            pd.DataFrame: User behavioral metrics
        """
        logger.info("Starting August user metrics generation...")
        print("🚀 Starting August user metrics generation...")
        
        try:
            # Step 1: Fetch data (only users with academic vertical interactions)
            print("📊 Step 1/3: Fetching data from august_feature (academic users only)...")
            df = self.fetch_august_data(max_users, academic_verticals_only=True)
            
            if df.empty:
                logger.warning("No data found")
                print("⚠️ No data found")
                return pd.DataFrame()
            
            # Step 2: Create user vectors
            print("🔄 Step 2/4: Creating user behavioral metrics...")
            user_metrics = self.create_user_vectors(df)
            
            if user_metrics.empty:
                logger.warning("No user metrics generated")
                print("⚠️ No user metrics generated")
                return pd.DataFrame()
            
            # Step 3: Calculate academic audience scores (no vertical filtering needed - already pre-filtered)
            print("🎓 Step 3/4: Calculating academic audience scores...")
            audience_scores = self.calculate_audience_scores(df, audience='אקדמאים', min_distinct_categories=1, filter_verticals=False)
            
            # Merge audience scores with user metrics
            if not audience_scores.empty:
                # Merge on user_unique_id
                user_metrics = user_metrics.merge(
                    audience_scores[['user_unique_id', 'audience', 'score_date', 'total_views', 'audience_views', 'score']], 
                    on='user_unique_id', 
                    how='left'
                )
                print(f"✅ Added audience scores for {len(audience_scores)} users")
            else:
                # Add empty audience columns
                user_metrics['audience'] = None
                user_metrics['score_date'] = None
                user_metrics['total_views'] = None
                user_metrics['audience_views'] = None
                user_metrics['score'] = None
                print("⚠️ No audience scores calculated")
            
            # Step 4: Save results
            print("💾 Step 4/4: Saving results...")
            user_metrics.to_csv(output_file, index=True)  # index=True to include user_unique_id
            print(f"✅ Results saved to: {output_file}")
            
            # Step 5: Append to BigQuery table if requested
            if append_to_table:
                print(f"📊 Step 5/5: Appending to BigQuery table...")
                self._append_to_bigquery_table(user_metrics, table_name)
            
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
            
            # Show academic audience score statistics
            if 'score' in user_metrics.columns and user_metrics['score'].notna().any():
                scored_users = user_metrics[user_metrics['score'].notna()]
                academic_users = scored_users[scored_users['score'] > 0]
                print(f"Users with academic scores: {len(scored_users):,}")
                print(f"Users with academic content: {len(academic_users):,}")
                print(f"Average academic score: {scored_users['score'].mean():.3f}")
                print(f"Max academic score: {scored_users['score'].max():.3f}")
                if len(academic_users) > 0:
                    print(f"Average score (academic users only): {academic_users['score'].mean():.3f}")
            
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
    
    def _append_to_bigquery_table(self, df: pd.DataFrame, table_name: str) -> None:
        """
        Append results to BigQuery table.
        
        Args:
            df: DataFrame with user metrics
            table_name: BigQuery table name
        """
        try:
            # Prepare table reference
            table_id = f"{self.project_id}.UsersClustering.{table_name}"
            
            # Reset index to make user_unique_id a column
            df_to_upload = df.reset_index()
            
            # Configure job
            job_config = bigquery.LoadJobConfig(
                write_disposition="WRITE_APPEND",  # Append mode
                create_disposition="CREATE_IF_NEEDED",  # Create table if doesn't exist
                autodetect=True  # Auto-detect schema
            )
            
            # Upload data
            print(f"📤 Uploading {len(df_to_upload):,} rows to {table_id}...")
            job = self.bq_client.load_table_from_dataframe(
                df_to_upload, 
                table_id, 
                job_config=job_config
            )
            
            # Wait for job to complete
            job.result()
            
            print(f"✅ Successfully appended {len(df_to_upload):,} rows to {table_id}")
            logger.info(f"Appended {len(df_to_upload)} rows to BigQuery table {table_id}")
            
        except Exception as e:
            logger.error(f"Error appending to BigQuery table: {e}")
            print(f"❌ Error appending to BigQuery table: {e}")
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
    
    # Ask about BigQuery table appending
    append_choice = input("Do you want to append results to BigQuery table 'users_with_audience_score'? (y/n): ").strip().lower()
    append_to_table = append_choice in ['y', 'yes']
    
    table_name = "users_with_audience_score"
    if append_to_table:
        custom_table = input(f"Enter custom table name (or press Enter for '{table_name}'): ").strip()
        table_name = custom_table if custom_table else table_name
    
    print(f"\nStarting metrics generation...")
    print(f"Max users: {max_users if max_users else 'All users'}")
    print(f"Output file: {output_file}")
    print(f"Append to BigQuery table: {'Yes' if append_to_table else 'No'}")
    if append_to_table:
        print(f"Table name: {table_name}")
    print("-" * 50)
    
    # Initialize generator and run analysis
    generator = AugustUserMetricsGenerator()
    results = generator.generate_metrics(max_users, output_file, append_to_table, table_name)
    
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
