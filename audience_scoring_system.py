#!/usr/bin/env python3
"""
Advanced Audience Scoring System

This script analyzes user behavior data to score users based on their relevance
to a target audience using Vertex AI. It uses a two-stage LLM approach:
1. Filter relevant verticals/categories for the audience
2. Score individual users based on their content consumption patterns

Usage:
    python audience_scoring_system.py

Or import as module:
    from audience_scoring_system import AudienceScorer
    scorer = AudienceScorer()
    results = scorer.score_audience("parents of babies", 30)
"""

import json
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from google.cloud import bigquery
import vertexai
from vertexai.generative_models import GenerativeModel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudienceScorer:
    """
    Advanced audience scoring system using Vertex AI for content relevance analysis.
    
    This class provides a streamlined approach to:
    1. Filter relevant content verticals/categories for target audiences
    2. Score users based on their content consumption patterns
    3. Generate comprehensive user behavioral profiles
    """
    
    def __init__(self, project_id: str = "wallabi-169712", location: str = "us-central1"):
        """
        Initialize the audience scorer.
        
        Args:
            project_id (str): Google Cloud Project ID
            location (str): Vertex AI location
        """
        self.project_id = project_id
        self.location = location
        self.bq_client = bigquery.Client(project=project_id)
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel("gemini-1.5-flash-002")
        
        logger.info(f"Initialized AudienceScorer for project: {project_id}")
    
    def _generate_text(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Generate text using Vertex AI Gemini model.
        
        Args:
            prompt (str): Input prompt
            temperature (float): Model temperature (0.0 for deterministic)
            
        Returns:
            str: Generated text response
        """
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": 2048,
            "top_p": 0.9,
            "top_k": 40
        }
        
        try:
            response = self.model.generate_content([prompt], generation_config=generation_config)
            return response.candidates[0].content.parts[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    def get_content_catalog(self, timeframe_days: int) -> Tuple[List[str], List[str]]:
        """
        Get all available verticals and categories from the data.
        
        Args:
            timeframe_days (int): Number of days to look back
            
        Returns:
            Tuple[List[str], List[str]]: Lists of verticals and categories
        """
        logger.info(f"Fetching content catalog for last {timeframe_days} days...")
        
        query = f"""
        SELECT DISTINCT 
            vertical_name,
            CategoryName
        FROM `{self.project_id}.UsersClustering.FeatureVectorAI`
        WHERE event_date > CURRENT_DATE() - INTERVAL {timeframe_days} DAY
          AND vertical_name IS NOT NULL 
          AND CategoryName IS NOT NULL
          AND vertical_name NOT IN ('וואלה', 'חדשות')
          AND CategoryName NOT LIKE '%חדשות%'
        ORDER BY vertical_name, CategoryName
        """
        
        try:
            df = self.bq_client.query(query).to_dataframe()
            verticals = df['vertical_name'].dropna().unique().tolist()
            categories = df['CategoryName'].dropna().unique().tolist()
            
            logger.info(f"Found {len(verticals)} verticals and {len(categories)} categories")
            return verticals, categories
            
        except Exception as e:
            logger.error(f"Error fetching content catalog: {e}")
            raise
    
    def filter_relevant_content(self, audience: str, verticals: List[str], categories: List[str]) -> Tuple[List[str], List[str]]:
        """
        Use LLM to filter relevant verticals and categories for the target audience.
        
        Args:
            audience (str): Target audience description
            verticals (List[str]): Available verticals
            categories (List[str]): Available categories
            
        Returns:
            Tuple[List[str], List[str]]: Filtered verticals and categories
        """
        logger.info(f"Filtering content for audience: '{audience}'")
        
        # Enhanced system prompt for better content filtering
        prompt = f"""
You are an expert content strategist and audience analyst specializing in Hebrew and Israeli content platforms.

TASK: Select the most relevant content verticals and categories for the target audience.

TARGET AUDIENCE: "{audience}"

AVAILABLE VERTICALS:
{json.dumps(verticals, ensure_ascii=False, indent=2)}

AVAILABLE CATEGORIES:
{json.dumps(categories, ensure_ascii=False, indent=2)}

SELECTION CRITERIA:
1. Choose verticals and categories that are semantically relevant to the target audience
2. Consider cultural context, Hebrew language nuances, and Israeli market specifics
3. Include both direct matches and related/adjacent content that the audience might consume
4. Prioritize content that indicates genuine interest or need from the target audience
5. Consider seasonal, demographic, and behavioral patterns

EXAMPLES:
- For "הורים לתינוקות" (parents of babies): Include בריאות, אוכל, משפחה, טיפים
- For "משקיעים צעירים" (young investors): Include כסף, קריירה, טכנולוגיה, נדל״ן
- For "סטודנטים" (students): Include חינוך, קריירה, כסף, אוכל, בידור

OUTPUT FORMAT:
Return ONLY a JSON object in this exact format:
{{
  "verticals": ["vertical1", "vertical2", ...],
  "categories": ["category1", "category2", ...]
}}

IMPORTANT:
- Select ONLY from the provided lists
- Do not invent or add content not in the lists
- Return only the JSON object, no additional text
- Be selective but comprehensive - aim for 3-8 verticals and 5-15 categories
"""
        
        try:
            response = self._generate_text(prompt, temperature=0.1)
            
            # Clean response
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Extract JSON
            first_brace = response.find("{")
            last_brace = response.rfind("}")
            if first_brace != -1 and last_brace != -1:
                response = response[first_brace:last_brace+1]
            
            result = json.loads(response)
            
            # Filter to only include items from original lists
            matched_verticals = [v for v in result.get("verticals", []) if v in verticals]
            matched_categories = [c for c in result.get("categories", []) if c in categories]
            
            logger.info(f"Selected {len(matched_verticals)} verticals and {len(matched_categories)} categories")
            logger.debug(f"Selected verticals: {matched_verticals}")
            logger.debug(f"Selected categories: {matched_categories}")
            
            return matched_verticals, matched_categories
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.error(f"Response: {response}")
            return [], []
        except Exception as e:
            logger.error(f"Error filtering content: {e}")
            return [], []
    
    def get_user_data(self, audience_verticals: List[str], audience_categories: List[str], 
                     timeframe_days: int, max_users: int = 10000) -> pd.DataFrame:
        """
        Fetch user interaction data based on filtered content.
        
        Args:
            audience_verticals (List[str]): Selected verticals
            audience_categories (List[str]): Selected categories
            timeframe_days (int): Timeframe in days
            max_users (int): Maximum number of users to fetch
            
        Returns:
            pd.DataFrame: User interaction data
        """
        logger.info(f"Fetching user data for {timeframe_days} days...")
        
        # Build content filter conditions
        verticals_str = ", ".join([f"'{v}'" for v in audience_verticals]) if audience_verticals else ""
        categories_str = ", ".join([f"'{c}'" for c in audience_categories]) if audience_categories else ""
        
        conditions = []
        if verticals_str:
            conditions.append(f"vertical_name IN ({verticals_str})")
        if categories_str:
            conditions.append(f"CategoryName IN ({categories_str})")
        
        if not conditions:
            logger.warning("No content filters applied")
            return pd.DataFrame()
        
        content_filter = " OR ".join(conditions)
        
        # First check data size
        count_query = f"""
        SELECT COUNT(DISTINCT user_unique_id) as user_count
        FROM `{self.project_id}.UsersClustering.FeatureVectorAI`
        WHERE event_date > CURRENT_DATE() - INTERVAL {timeframe_days} DAY
          AND ({content_filter})
        """
        
        count_df = self.bq_client.query(count_query).to_dataframe()
        total_users = count_df['user_count'].iloc[0]
        logger.info(f"Found {total_users:,} users matching content criteria")
        
        # Fetch user data with sampling if needed
        if total_users > max_users:
            sample_percent = (max_users / total_users) * 100
            logger.info(f"Sampling {sample_percent:.2f}% of users ({max_users:,} users)")
            
            query = f"""
            SELECT 
                user_unique_id,
                event_date,
                vertical_name,
                CategoryName,
                page_title,
                item_title,
                hour_of_day,
                day_of_week
            FROM `{self.project_id}.UsersClustering.FeatureVectorAI`
            WHERE event_date > CURRENT_DATE() - INTERVAL {timeframe_days} DAY
              AND ({content_filter})
              AND RAND() < {sample_percent/100}
            ORDER BY user_unique_id, event_date
            """
        else:
            query = f"""
            SELECT 
                user_unique_id,
                event_date,
                vertical_name,
                CategoryName,
                page_title,
                item_title,
                hour_of_day,
                day_of_week
            FROM `{self.project_id}.UsersClustering.FeatureVectorAI`
            WHERE event_date > CURRENT_DATE() - INTERVAL {timeframe_days} DAY
              AND ({content_filter})
            ORDER BY user_unique_id, event_date
            """
        
        try:
            df = self.bq_client.query(query).to_dataframe()
            logger.info(f"Fetched {len(df):,} interactions for {df['user_unique_id'].nunique():,} users")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching user data: {e}")
            raise
    
    def create_user_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive user behavioral profiles.
        
        Args:
            df (pd.DataFrame): User interaction data
            
        Returns:
            pd.DataFrame: User profiles with behavioral metrics
        """
        logger.info("Creating user behavioral profiles...")
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert event_date to datetime
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        # Calculate behavioral metrics per user
        user_profiles = []
        
        for user_id, user_data in df.groupby('user_unique_id'):
            # Get distinct content (remove duplicates)
            distinct_page_titles = user_data['page_title'].dropna().unique().tolist()
            distinct_item_titles = user_data['item_title'].dropna().unique().tolist()
            
            profile = {
                'user_unique_id': user_id,
                'total_interactions': len(user_data),
                'unique_verticals': user_data['vertical_name'].nunique(),
                'unique_categories': user_data['CategoryName'].nunique(),
                'unique_pages': len(distinct_page_titles),
                'unique_items': len(distinct_item_titles),
                'total_distinct_articles': len(distinct_page_titles) + len(distinct_item_titles),
                'date_span_days': (user_data['event_date'].max() - user_data['event_date'].min()).days + 1,
                'weekend_activity_ratio': (user_data['day_of_week'].isin([5, 6]).sum() / len(user_data)),
                'evening_activity_ratio': (user_data['hour_of_day'].between(18, 23).sum() / len(user_data)),
                'morning_activity_ratio': (user_data['hour_of_day'].between(6, 11).sum() / len(user_data)),
                'content_diversity_score': user_data['vertical_name'].nunique() / len(user_data),
                'engagement_intensity': len(user_data) / ((user_data['event_date'].max() - user_data['event_date'].min()).days + 1),
                'page_titles': distinct_page_titles,  # Already distinct
                'item_titles': distinct_item_titles,  # Already distinct
                'vertical_distribution': user_data['vertical_name'].value_counts().to_dict(),
                'category_distribution': user_data['CategoryName'].value_counts().to_dict()
            }
            user_profiles.append(profile)
        
        profiles_df = pd.DataFrame(user_profiles)
        logger.info(f"Created profiles for {len(profiles_df)} users")
        
        return profiles_df
    
    def score_users_with_llm(self, user_profiles: pd.DataFrame, audience: str, 
                           sample_size: int = 50) -> pd.DataFrame:
        """
        Score users using LLM based on their content consumption patterns.
        Focuses on content relevance (page_title, item_title, vertical_name, CategoryName).
        
        Args:
            user_profiles (pd.DataFrame): User behavioral profiles
            audience (str): Target audience description
            sample_size (int): Number of users to score
            
        Returns:
            pd.DataFrame: Users with relevance scores
        """
        logger.info(f"Scoring {min(sample_size, len(user_profiles))} users for audience: '{audience}'")
        
        # Sample users if needed
        if len(user_profiles) > sample_size:
            sample_users = user_profiles.sample(n=sample_size, random_state=42)
        else:
            sample_users = user_profiles.copy()
        
        scored_users = []
        
        for idx, user in sample_users.iterrows():
            try:
                # Get distinct content (remove duplicates)
                distinct_page_titles = list(set(user['page_titles']))[:25]  # Limit for prompt size
                distinct_item_titles = list(set(user['item_titles']))[:25]  # Limit for prompt size
                distinct_verticals = list(user['vertical_distribution'].keys())
                distinct_categories = list(user['category_distribution'].keys())
                
                # Content-focused scoring prompt
                prompt = f"""
You are an expert content analyst specializing in Hebrew/Israeli digital content relevance.

TASK: Score this user's relevance to the target audience based ONLY on the content they consumed.

TARGET AUDIENCE: "{audience}"

USER'S DISTINCT CONTENT CONSUMPTION:
Distinct Page Titles (Articles Read): {distinct_page_titles}
Distinct Item Titles (Articles Read): {distinct_item_titles}
Content Verticals: {distinct_verticals}
Content Categories: {distinct_categories}

CONTENT STATISTICS:
- Total Distinct Articles Read: {len(distinct_page_titles) + len(distinct_item_titles)}
- Content Verticals Engaged: {len(distinct_verticals)}
- Content Categories Engaged: {len(distinct_categories)}

SCORING FOCUS:
Score based ONLY on content relevance - how well the articles, verticals, and categories align with the target audience's interests.

SCORING CRITERIA:
1. Article Relevance (70%): How many of the distinct page_title and item_title articles are relevant to the target audience?
2. Vertical/Category Relevance (30%): How well do the verticals and categories match the target audience?

SCORING LOGIC:
- Count how many distinct articles are relevant to the target audience
- Calculate the ratio: (relevant articles / total distinct articles)
- Consider vertical and category alignment
- Example: If user read 7 distinct articles and 5 are highly relevant to "הורים לתינוקות" → high score (0.8+)

SCORING SCALE:
- 0.9-1.0: Excellent match - most/all articles are highly relevant to target audience
- 0.7-0.8: Good match - majority of articles are relevant to target audience  
- 0.5-0.6: Moderate match - some articles are relevant to target audience
- 0.3-0.4: Weak match - few articles are relevant to target audience
- 0.0-0.2: Poor match - minimal or no articles relevant to target audience

CULTURAL CONTEXT:
Consider Hebrew language nuances, Israeli cultural references, and local market specifics when evaluating article relevance.

IMPORTANT: 
- Focus on content relevance, NOT reading patterns, timing, or engagement frequency
- Consider the semantic meaning of article titles, not just keywords
- A user who reads 5 highly relevant articles should score higher than someone who reads 20 marginally relevant articles

OUTPUT: Return only a single number between 0.0 and 1.0 (e.g., 0.75). No explanation or additional text.
"""
                
                score_text = self._generate_text(prompt, temperature=0.1)
                
                # Parse score
                try:
                    score = float(score_text.strip())
                    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                except ValueError:
                    logger.warning(f"Could not parse score: {score_text}")
                    score = 0.5  # Default neutral score
                
                # Add score to user profile
                user_with_score = user.copy()
                user_with_score['relevance_score'] = score
                user_with_score['audience'] = audience
                user_with_score['scored_at'] = datetime.now().isoformat()
                user_with_score['distinct_articles_count'] = len(distinct_page_titles) + len(distinct_item_titles)
                user_with_score['distinct_verticals_count'] = len(distinct_verticals)
                user_with_score['distinct_categories_count'] = len(distinct_categories)
                
                scored_users.append(user_with_score)
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error scoring user {user['user_unique_id']}: {e}")
                continue
        
        if scored_users:
            result_df = pd.DataFrame(scored_users)
            logger.info(f"Successfully scored {len(result_df)} users")
            return result_df
        else:
            logger.warning("No users were successfully scored")
            return pd.DataFrame()
    
    def score_audience(self, audience: str, timeframe_days: int = 30, 
                      max_users: int = 10000, sample_size: int = 50) -> pd.DataFrame:
        """
        Main method to score users for a target audience.
        
        Args:
            audience (str): Target audience description
            timeframe_days (int): Number of days to look back
            max_users (int): Maximum users to fetch from database
            sample_size (int): Number of users to score with LLM
            
        Returns:
            pd.DataFrame: Scored users with behavioral profiles
        """
        logger.info(f"Starting audience scoring for: '{audience}' ({timeframe_days} days)")
        
        try:
            # Step 1: Get content catalog
            verticals, categories = self.get_content_catalog(timeframe_days)
            
            # Step 2: Filter relevant content
            relevant_verticals, relevant_categories = self.filter_relevant_content(
                audience, verticals, categories
            )
            
            if not relevant_verticals and not relevant_categories:
                logger.warning("No relevant content found for audience")
                return pd.DataFrame()
            
            # Step 3: Get user data
            user_data = self.get_user_data(
                relevant_verticals, relevant_categories, timeframe_days, max_users
            )
            
            if user_data.empty:
                logger.warning("No user data found")
                return pd.DataFrame()
            
            # Step 4: Create user profiles
            user_profiles = self.create_user_profiles(user_data)
            
            # Step 5: Score users with LLM
            scored_users = self.score_users_with_llm(user_profiles, audience, sample_size)
            
            # Add metadata
            scored_users['timeframe_days'] = timeframe_days
            scored_users['relevant_verticals'] = str(relevant_verticals)
            scored_users['relevant_categories'] = str(relevant_categories)
            
            logger.info(f"Audience scoring complete! Scored {len(scored_users)} users")
            return scored_users
            
        except Exception as e:
            logger.error(f"Error in audience scoring: {e}")
            raise


def main():
    """Interactive command-line interface."""
    print("=== Advanced Audience Scoring System ===")
    print("Score users based on their content consumption relevance to target audiences")
    print()
    
    # Get user inputs
    audience = input("Enter target audience (e.g., 'הורים לתינוקות', 'משקיעים צעירים'): ").strip()
    
    try:
        timeframe = int(input("Enter timeframe in days (default: 30): ").strip() or "30")
    except ValueError:
        timeframe = 30
    
    try:
        max_users = int(input("Enter max users to fetch (default: 10000): ").strip() or "10000")
    except ValueError:
        max_users = 10000
    
    try:
        sample_size = int(input("Enter sample size for scoring (default: 50): ").strip() or "50")
    except ValueError:
        sample_size = 50
    
    output_file = input("Enter output CSV file (default: audience_scores.csv): ").strip() or "audience_scores.csv"
    
    print(f"\nStarting analysis...")
    print(f"Audience: {audience}")
    print(f"Timeframe: {timeframe} days")
    print(f"Max users: {max_users:,}")
    print(f"Sample size: {sample_size}")
    print("-" * 50)
    
    # Initialize scorer and run analysis
    scorer = AudienceScorer()
    results = scorer.score_audience(audience, timeframe, max_users, sample_size)
    
    if results.empty:
        print("No results generated. Please check your inputs and try again.")
        return
    
    # Save results
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"Scored {len(results)} users")
    
    # Show summary
    print(f"\nScore Distribution:")
    print(f"Average Score: {results['relevance_score'].mean():.3f}")
    print(f"High Relevance (>0.7): {len(results[results['relevance_score'] > 0.7])} users")
    print(f"Medium Relevance (0.4-0.7): {len(results[(results['relevance_score'] >= 0.4) & (results['relevance_score'] <= 0.7)])} users")
    print(f"Low Relevance (<0.4): {len(results[results['relevance_score'] < 0.4])} users")
    
    # Show top users
    print(f"\nTop 5 Most Relevant Users:")
    top_users = results.nlargest(5, 'relevance_score')[['user_unique_id', 'relevance_score', 'total_interactions', 'unique_verticals']]
    print(top_users.to_string(index=False))
    
    return results


if __name__ == "__main__":
    results = main()
