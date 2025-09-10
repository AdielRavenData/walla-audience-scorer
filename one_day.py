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
import re
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
        self.src_table = "september_first_feature_vector_ai_distinct"
        
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
    
    def get_content_catalog(self, target_date: str) -> Tuple[List[str], List[str]]:
        """
        Get all available verticals and categories from the data for a specific date.
        
        Args:
            target_date (str): Target date in YYYY-MM-DD format
            
        Returns:
            Tuple[List[str], List[str]]: Lists of verticals and categories
        """
        logger.info(f"Fetching content catalog for date: {target_date}")
        
        query = f"""
        SELECT DISTINCT 
            vertical_name,
            CategoryName
        FROM `{self.project_id}.UsersClustering.{self.src_table}`
        WHERE event_date = '{target_date}'
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
                     target_date: str, max_users: int = 10000) -> pd.DataFrame:
        """
        Fetch user interaction data based on filtered content for a specific date.
        
        Args:
            audience_verticals (List[str]): Selected verticals
            audience_categories (List[str]): Selected categories
            target_date (str): Target date in YYYY-MM-DD format
            max_users (int): Maximum number of users to fetch
            
        Returns:
            pd.DataFrame: User interaction data
        """
        logger.info(f"Fetching user data for date: {target_date}")
        
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
        FROM `{self.project_id}.UsersClustering.{self.src_table}`
        WHERE event_date = '{target_date}'
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
            FROM `{self.project_id}.UsersClustering.{self.src_table}`
            WHERE event_date = '{target_date}'
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
            FROM `{self.project_id}.UsersClustering.{self.src_table}`
            WHERE event_date = '{target_date}'
              AND ({content_filter})
            ORDER BY user_unique_id, event_date
            """
        
        try:
            print("⏳ Executing BigQuery query... (this may take a few minutes)")
            df = self.bq_client.query(query).to_dataframe()
            logger.info(f"Fetched {len(df):,} interactions for {df['user_unique_id'].nunique():,} users")
            print(f"✅ BigQuery query completed! Fetched {len(df):,} interactions for {df['user_unique_id'].nunique():,} users")
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
        print("🔄 Creating user behavioral profiles...")
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert event_date to datetime
        df['event_date'] = pd.to_datetime(df['event_date'])
        
        # Calculate behavioral metrics per user
        user_profiles = []
        total_users = df['user_unique_id'].nunique()
        processed_users = 0
        
        print(f"📊 Processing {total_users} users for behavioral profiles...")
        
        for user_id, user_data in df.groupby('user_unique_id'):
            processed_users += 1
            if processed_users % 1000 == 0:
                print(f"🔄 Processed {processed_users}/{total_users} users for profiles...")
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
                'content_diversity_score': user_data['vertical_name'].nunique() / len(user_data),
                'page_titles': distinct_page_titles,  # Already distinct
                'item_titles': distinct_item_titles,  # Already distinct
                'vertical_distribution': user_data['vertical_name'].value_counts().to_dict(),
                'category_distribution': user_data['CategoryName'].value_counts().to_dict()
            }
            user_profiles.append(profile)
        
        profiles_df = pd.DataFrame(user_profiles)
        logger.info(f"Created profiles for {len(profiles_df)} users")
        print(f"✅ Created profiles for {len(profiles_df)} users")
        
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
        print(f"🤖 Starting LLM scoring for {min(sample_size, len(user_profiles))} users...")
        
        # Sample users if needed
        if len(user_profiles) > sample_size:
            sample_users = user_profiles.sample(n=sample_size, random_state=42)
        else:
            sample_users = user_profiles.copy()
        
        scored_users = []
        total_users = len(sample_users)
        print(f"📊 Will score {total_users} users with LLM...")
        
        for idx, user in sample_users.iterrows():
            start_time = time.time()
            try:
                # Get distinct content (remove duplicates)
                distinct_page_titles = list(set(user['page_titles']))[:25]  # Limit for prompt size
                distinct_item_titles = list(set(user['item_titles']))[:25]  # Limit for prompt size
                distinct_verticals = list(user['vertical_distribution'].keys())
                distinct_categories = list(user['category_distribution'].keys())
                
                # Content-focused scoring prompt (shortened for performance)
                prompt = f"""
Score user relevance to target audience: "{audience}"

USER CONTENT:
Articles: {distinct_page_titles[:10]} {distinct_item_titles[:10]}
Verticals: {distinct_verticals}
Categories: {distinct_categories}

SCORING (50% articles + 50% verticals/categories):
- Article Score: How many articles match the audience?
- Vertical/Category Score: How well do verticals/categories match?
- Final = (Article × 0.5) + (Vertical/Category × 0.5)

SCALE (0.6-1.0 only):
- 0.9-1.0: Excellent match
- 0.8-0.89: Good match  
- 0.7-0.79: Moderate match
- 0.6-0.69: Basic match

CRITICAL: Return ONLY a single number between 0.6 and 1.0. No text, no explanation, no analysis. Just the number.
Example: 0.75
"""
                
                score_text = self._generate_text(prompt, temperature=0.1)
                
                # Parse score - extract number from response
                try:
                    # Clean the response and extract the first number
                    # Look for decimal numbers in the response
                    numbers = re.findall(r'\d+\.?\d*', score_text.strip())
                    if numbers:
                        score = float(numbers[0])
                        score = max(0.6, min(1.0, score))  # Clamp to [0.6, 1.0]
                    else:
                        # If no number found, use default
                        score = 0.7
                        logger.warning(f"No number found in response: {score_text[:100]}...")
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse score: {score_text[:100]}...")
                    score = 0.7  # Default score for filtered content
                
                # Add score to user profile
                user_with_score = user.copy()
                user_with_score['relevance_score'] = score
                user_with_score['audience'] = audience
                user_with_score['scored_at'] = datetime.now().isoformat()
                
                scored_users.append(user_with_score)
                
                # Log progress with timing
                end_time = time.time()
                duration = end_time - start_time
                progress = len(scored_users)
                logger.info(f"✅ User {progress}/{total_users} scored: {user['user_unique_id']} -> {score:.3f} (took {duration:.2f}s)")
                print(f"✅ User {progress}/{total_users} scored: {user['user_unique_id']} -> {score:.3f} (took {duration:.2f}s)")
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                progress = len(scored_users) + 1
                logger.error(f"❌ User {progress}/{total_users} failed: {user['user_unique_id']} (took {duration:.2f}s) - Error: {e}")
                print(f"❌ User {progress}/{total_users} failed: {user['user_unique_id']} (took {duration:.2f}s) - Error: {e}")
                continue
        
        if scored_users:
            result_df = pd.DataFrame(scored_users)
            logger.info(f"Successfully scored {len(result_df)} users")
            return result_df
        else:
            logger.warning("No users were successfully scored")
            return pd.DataFrame()
    
    def score_audience(self, audience: str, target_date: str, 
                      max_users: int = 10000, sample_size: int = 50) -> pd.DataFrame:
        """
        Main method to score users for a target audience on a specific date.
        
        Args:
            audience (str): Target audience description
            target_date (str): Target date in YYYY-MM-DD format
            max_users (int): Maximum users to fetch from database
            sample_size (int): Number of users to score with LLM
            
        Returns:
            pd.DataFrame: Scored users with behavioral profiles
        """
        logger.info(f"Starting audience scoring for: '{audience}' on {target_date}")
        print(f"🚀 Starting audience scoring for: '{audience}' on {target_date}")
        
        try:
            # Step 1: Get content catalog
            print("📊 Step 1/5: Getting content catalog...")
            verticals, categories = self.get_content_catalog(target_date)
            
            # Step 2: Filter relevant content
            print("🔍 Step 2/5: Filtering relevant content with LLM...")
            relevant_verticals, relevant_categories = self.filter_relevant_content(
                audience, verticals, categories
            )
            
            if not relevant_verticals and not relevant_categories:
                logger.warning("No relevant content found for audience")
                print("⚠️ No relevant content found for audience")
                return pd.DataFrame()
            
            # Step 3: Get user data
            print("📥 Step 3/5: Fetching user data from BigQuery...")
            user_data = self.get_user_data(
                relevant_verticals, relevant_categories, target_date, max_users
            )
            
            if user_data.empty:
                logger.warning("No user data found")
                print("⚠️ No user data found")
                return pd.DataFrame()
            
            # Step 4: Create user profiles
            print("👥 Step 4/5: Creating user behavioral profiles...")
            user_profiles = self.create_user_profiles(user_data)
            
            # Step 5: Score users with LLM
            print("🤖 Step 5/5: Starting LLM scoring for all users...")
            scored_users = self.score_users_with_llm(user_profiles, audience, sample_size)
            
            # Add metadata
            scored_users['target_date'] = target_date
            
            logger.info(f"Audience scoring complete! Scored {len(scored_users)} users")
            return scored_users
            
        except Exception as e:
            logger.error(f"Error in audience scoring: {e}")
            raise


def main():
    """Interactive command-line interface."""
    print("=== One-Day Audience Scoring System ===")
    print("Score users based on their content consumption relevance to target audiences for a specific date")
    print()
    
    # Get user inputs
    audience = input("Enter target audience (e.g., 'הורים לתינוקות', 'משקיעים צעירים'): ").strip()
    
    target_date = input("Enter target date (YYYY-MM-DD, default: 2025-09-01): ").strip() or "2025-09-01"
    
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
    print(f"Target Date: {target_date}")
    print(f"Max users: {max_users:,}")
    print(f"Sample size: {sample_size}")
    print("-" * 50)
    
    # Initialize scorer and run analysis
    scorer = AudienceScorer()
    results = scorer.score_audience(audience, target_date, max_users, sample_size)
    
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