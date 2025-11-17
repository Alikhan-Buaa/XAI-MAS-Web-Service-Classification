import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from pathlib import Path

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def return_json_from_txt(txt_path):
    '''return the json data for the given txt file'''
    with open(txt_path, encoding='utf-8') as file:
        raw = file.read()
        data = json.loads(raw, strict=False)
    return data

def extract_web_service_description_category(data, desc_key='Description', cat_key='PrimaryCategory'):
    '''extract the columns from given pandas dataframe'''
    df = pd.DataFrame(data)
    df_final = pd.DataFrame(df[[desc_key, cat_key]])
    df_final.columns = ['Service Description', 'Service Classification']
    return df_final

def remove_duplicates(df, desc_column='Service Description'):
    '''Remove duplicate entries based on service description'''
    print(f"Original dataset size: {len(df)}")
    df_clean = df.drop_duplicates(subset=[desc_column], keep='first')
    duplicates_removed = len(df) - len(df_clean)
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"Clean dataset size: {len(df_clean)}")
    return df_clean

def calculate_text_statistics(text):
    '''Calculate word count and character length for a text'''
    if pd.isna(text):
        return 0, 0
    
    text_str = str(text)
    words = word_tokenize(text_str.lower())
    word_count = len(words)
    char_length = len(text_str)
    
    return word_count, char_length

def get_top_words(texts, top_n=10):
    '''Get top N most frequent words from a list of texts'''
    stop_words = set(stopwords.words('english'))
    all_words = []
    
    for text in texts:
        if pd.isna(text):
            continue
        words = word_tokenize(str(text).lower())
        # Filter out stop words and punctuation
        filtered_words = [word for word in words if word.isalnum() and word not in stop_words and len(word) > 2]
        all_words.extend(filtered_words)
    
    word_freq = Counter(all_words)
    return word_freq.most_common(top_n)

def analyze_category_statistics(df, category_column='Service Classification', desc_column='Service Description'):
    '''Analyze statistics for each category'''
    categories = df[category_column].unique()
    stats_data = []
    
    print("\n=== CATEGORY ANALYSIS ===")
    for category in categories:
        category_df = df[df[category_column] == category]
        texts = category_df[desc_column].tolist()
        
        # Calculate word counts and character lengths
        word_counts = []
        char_lengths = []
        
        for text in texts:
            word_count, char_length = calculate_text_statistics(text)
            word_counts.append(word_count)
            char_lengths.append(char_length)
        
        # Get top words for this category
        top_words = get_top_words(texts, top_n=10)
        top_words_str = ', '.join([f"{word}({count})" for word, count in top_words])
        
        # Calculate statistics
        stats = {
            'Category': category,
            'Sample_Count': len(category_df),
            'Word_Count_Min': min(word_counts) if word_counts else 0,
            'Word_Count_Max': max(word_counts) if word_counts else 0,
            'Word_Count_Avg': np.mean(word_counts) if word_counts else 0,
            'Char_Length_Min': min(char_lengths) if char_lengths else 0,
            'Char_Length_Max': max(char_lengths) if char_lengths else 0,
            'Char_Length_Avg': np.mean(char_lengths) if char_lengths else 0,
            'Top_10_Words': top_words_str
        }
        
        stats_data.append(stats)
        
        # Print category analysis
        print(f"\nCategory: {category}")
        print(f"  Samples: {stats['Sample_Count']}")
        print(f"  Words: Min={stats['Word_Count_Min']}, Max={stats['Word_Count_Max']}, Avg={stats['Word_Count_Avg']:.1f}")
        print(f"  Chars: Min={stats['Char_Length_Min']}, Max={stats['Char_Length_Max']}, Avg={stats['Char_Length_Avg']:.1f}")
        print(f"  Top words: {', '.join([word for word, count in top_words[:5]])}")
    
    return pd.DataFrame(stats_data)

def filter_top_n_web_service_categories(df, label_column='Service Classification', top_n=50):
    '''filter data frame based on top_n category'''
    label_counts = df[label_column].value_counts()
    top_labels = label_counts.head(top_n).index
    filtered_df = df[df[label_column].isin(top_labels)]
    return filtered_df, label_counts[top_labels]

def create_rank_based_dataset(df, label_column='Service Classification', start_rank=1, end_rank=10):
    '''Create dataset for specific rank range (e.g., rank 1-10, 11-20, etc.)'''
    label_counts = df[label_column].value_counts()
    
    # Get labels for the specific rank range
    rank_labels = label_counts.iloc[start_rank-1:end_rank].index
    filtered_df = df[df[label_column].isin(rank_labels)]
    
    return filtered_df, label_counts[rank_labels]

def save_to_csv(df, output_path):
    '''data frame to csv file saving'''
    print(output_path)
    df.to_csv(output_path, encoding='utf-8', index=False, header=True)

def plot_web_services_category_distribution(label_counts, title, output_path):
    '''Plot category distribution'''
    colors = sns.color_palette("viridis", len(label_counts))
    
    plt.figure(figsize=(20, max(12, len(label_counts) * 0.5)))
    ax = label_counts.sort_values().plot(
        kind='barh',
        color=colors,
        fontsize=12,
        edgecolor='black'
    )
    
    # Add count labels at end of each bar
    for bar in ax.patches:
        plt.text(bar.get_width() + max(label_counts.values) * 0.01,
                 bar.get_y() + bar.get_height() / 2,
                 str(int(bar.get_width())),
                 va='center',
                 fontsize=11,
                 color='black')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Count", fontsize=12)
    plt.ylabel("Categories", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    #plt.show()
    plt.close()

def process_comprehensive_web_services_analysis(txt_path, output_dir, top_n_list):
    '''Main processing function with comprehensive analysis'''
    
    # Load and prepare data
    print("Loading data...")
    data = return_json_from_txt(txt_path)
    df = extract_web_service_description_category(data)
    
    # Remove duplicates
    print("\nRemoving duplicates...")
    df_clean = remove_duplicates(df)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save clean dataset
    clean_data_path = os.path.join(output_dir, "clean_dataset.csv")
    save_to_csv(df_clean, clean_data_path)
    print(f"Clean dataset saved to: {clean_data_path}")
    
    # Process each top_n value
    for n in top_n_list:
        print(f"\n{'='*50}")
        print(f"Processing Top {n} Categories")
        print(f"{'='*50}")
        
        # Create cumulative top-n dataset
        filtered_df, label_counts = filter_top_n_web_service_categories(df_clean, top_n=n)
        
        # Save cumulative dataset
        output_cumulative_dir = Path("data/analysis") / "cumulative"

        # Create the directory (with parents, so no error if output_dir doesn't exist yet)
        output_cumulative_dir.mkdir(parents=True, exist_ok=True)

        csv_path = os.path.join(output_cumulative_dir, f"Top_{n}_Categories_Cumulative.csv")
        plot_path = os.path.join(output_cumulative_dir, f"Top_{n}_Categories_Cumulative.png")
        
        print(f"Cumulative Top {n} - Category Distribution:")
        print(filtered_df["Service Classification"].value_counts())
        
        save_to_csv(filtered_df, csv_path)
        plot_web_services_category_distribution(label_counts, f"Top {n} Web Service Categories (Cumulative)", plot_path)
        
        # Analyze statistics for cumulative dataset
        stats_df = analyze_category_statistics(filtered_df)
        stats_path = os.path.join(output_cumulative_dir, f"Top_{n}_Categories_Cumulative_Statistics.csv")
        save_to_csv(stats_df, stats_path)
        
        print(f"Saved Top {n} Cumulative: CSV, plot, and statistics")
        
        # Create rank-based datasets for all ranges
        if n == 10:
            # Special case for rank 1-10 (first range)
            print(f"\nCreating Rank-based dataset: Rank 1 to {n}")
            rank_df, rank_label_counts = create_rank_based_dataset(
                df_clean, start_rank=1, end_rank=n
            )
            
            # Save rank-based dataset
            output_category_dir = Path("data/analysis") / "category-wise"
            rank_csv_path = os.path.join(output_category_dir, f"Rank_1_to_{n}_Categories.csv")
            rank_plot_path = os.path.join(output_category_dir, f"Rank_1_to_{n}_Categories.png")
            
            print(f"Rank 1-{n} - Category Distribution:")
            print(rank_df["Service Classification"].value_counts())
            
            save_to_csv(rank_df, rank_csv_path)
            plot_web_services_category_distribution(
                rank_label_counts, 
                f"Categories Rank 1 to {n}", 
                rank_plot_path
            )
            
            # Analyze statistics for rank-based dataset
            rank_stats_df = analyze_category_statistics(rank_df)
            rank_stats_path = os.path.join(output_category_dir, f"Rank_1_to_{n}_Categories_Statistics.csv")
            save_to_csv(rank_stats_df, rank_stats_path)
            
            print(f"Saved Rank 1-{n}: CSV, plot, and statistics")
            
        elif n > 10:
            # For ranges 11-20, 21-30, etc.
            prev_n = top_n_list[top_n_list.index(n) - 1] if top_n_list.index(n) > 0 else 0
            
            print(f"\nCreating Rank-based dataset: Rank {prev_n+1} to {n}")
            rank_df, rank_label_counts = create_rank_based_dataset(
                df_clean, start_rank=prev_n+1, end_rank=n
            )
            
            # Save rank-based dataset
            rank_csv_path = os.path.join(output_category_dir, f"Rank_{prev_n+1}_to_{n}_Categories.csv")
            rank_plot_path = os.path.join(output_category_dir, f"Rank_{prev_n+1}_to_{n}_Categories.png")
            
            print(f"Rank {prev_n+1}-{n} - Category Distribution:")
            print(rank_df["Service Classification"].value_counts())
            
            save_to_csv(rank_df, rank_csv_path)
            plot_web_services_category_distribution(
                rank_label_counts, 
                f"Categories Rank {prev_n+1} to {n}", 
                rank_plot_path
            )
            
            # Analyze statistics for rank-based dataset
            rank_stats_df = analyze_category_statistics(rank_df)
            rank_stats_path = os.path.join(output_category_dir, f"Rank_{prev_n+1}_to_{n}_Categories_Statistics.csv")
            save_to_csv(rank_stats_df, rank_stats_path)
            
            print(f"Saved Rank {prev_n+1}-{n}: CSV, plot, and statistics")
    
    # Create overall summary statistics
    print(f"\n{'='*50}")
    print("Creating Overall Summary")
    print(f"{'='*50}")
    
    overall_stats = analyze_category_statistics(df_clean)
    overall_stats_path = os.path.join(output_dir, "Overall_Dataset_Statistics.csv")
    save_to_csv(overall_stats, overall_stats_path)
    
    # Create summary report
    create_summary_report(df_clean, output_dir, top_n_list)
    
    print(f"\nAnalysis complete! All files saved in: {output_dir}")

def create_summary_report(df, output_dir, top_n_list):
    '''Create a comprehensive summary report'''
    
    summary_data = {
        'Metric': [],
        'Value': []
    }
    
    # Overall dataset metrics
    summary_data['Metric'].extend([
        'Total Samples',
        'Total Categories',
        'Average Samples per Category',
        'Most Common Category',
        'Most Common Category Count',
        'Least Common Category',
        'Least Common Category Count'
    ])
    
    category_counts = df['Service Classification'].value_counts()
    
    summary_data['Value'].extend([
        len(df),
        len(category_counts),
        f"{len(df) / len(category_counts):.1f}",
        category_counts.index[0],
        category_counts.iloc[0],
        category_counts.index[-1],
        category_counts.iloc[-1]
    ])
    
    # Add top-n coverage information
    for n in top_n_list:
        top_n_coverage = (category_counts.head(n).sum() / len(df)) * 100
        summary_data['Metric'].append(f'Top {n} Categories Coverage (%)')
        summary_data['Value'].append(f"{top_n_coverage:.1f}%")
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "Dataset_Summary_Report.csv")
    save_to_csv(summary_df, summary_path)
    
    print(f"Summary report saved to: {summary_path}")

if __name__ == "__main__":
    input_txt = "data/raw/ProgrammWebScrapy.txt"
    output_dir = "data/analysis"
    top_web_services_category_count_list = [10, 20, 30, 40, 50]
    
    process_comprehensive_web_services_analysis(
        input_txt, 
        output_dir, 
        top_web_services_category_count_list

    )