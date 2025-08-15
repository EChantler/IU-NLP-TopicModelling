import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List, Dict
import os

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def plot_size_scaling(
    dataset_percentages: List[float],
    model_performance: Dict[str, Dict[str, List[float]]],
    save_path: str = './results/dataset_size_scaling.png',
    title: str = 'Model Performance vs Dataset Size'
):
    """
    Plot model performance (accuracy and F1-score) vs dataset size.
    
    Args:
        dataset_percentages: List of dataset size percentages (e.g., [10, 25, 50, 75, 100])
        model_performance: Dictionary with structure:
            {
                'model_name': {
                    'accuracy': [list of accuracy scores],
                    'f1_score': [list of f1 scores]
                }
            }
        save_path: Path to save the plot
        title: Title for the plot
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors and markers for different models
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    # Plot F1-Score vs Dataset Size
    for i, (model_name, metrics) in enumerate(model_performance.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax1.plot(dataset_percentages, metrics['f1_score'], 
                marker=marker, color=color, linewidth=2.5, markersize=8,
                label=model_name.replace('_', ' ').title(), alpha=0.8)
        
        # Add value labels on points
        for x, y in zip(dataset_percentages, metrics['f1_score']):
            ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, alpha=0.7)
    
    ax1.set_xlabel('Dataset Size (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score (Macro)', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score vs Dataset Size', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.set_ylim(0, 1)
    
    # Plot Accuracy vs Dataset Size
    for i, (model_name, metrics) in enumerate(model_performance.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        ax2.plot(dataset_percentages, metrics['accuracy'], 
                marker=marker, color=color, linewidth=2.5, markersize=8,
                label=model_name.replace('_', ' ').title(), alpha=0.8)
        
        # Add value labels on points
        for x, y in zip(dataset_percentages, metrics['accuracy']):
            ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=9, alpha=0.7)
    
    ax2.set_xlabel('Dataset Size (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy vs Dataset Size', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.set_ylim(0, 1)
    
    # Add overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Dataset size scaling plot saved to: {save_path}")

def plot_improvement_rates(
    dataset_percentages: List[float],
    model_performance: Dict[str, Dict[str, List[float]]],
    save_path: str = './results/performance_improvement_rates.png'
):
    """
    Plot the rate of improvement for each model as dataset size increases.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i, (model_name, metrics) in enumerate(model_performance.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # Calculate improvement rates (difference between consecutive points)
        f1_improvements = np.diff(metrics['f1_score'])
        acc_improvements = np.diff(metrics['accuracy'])
        improvement_x = dataset_percentages[1:]  # Skip first point for diff
        
        ax1.plot(improvement_x, f1_improvements, 
                marker=marker, color=color, linewidth=2.5, markersize=8,
                label=model_name.replace('_', ' ').title(), alpha=0.8)
        
        ax2.plot(improvement_x, acc_improvements, 
                marker=marker, color=color, linewidth=2.5, markersize=8,
                label=model_name.replace('_', ' ').title(), alpha=0.8)
    
    ax1.set_xlabel('Dataset Size (%)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score Improvement', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score Improvement Rate', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Dataset Size (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Improvement', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Improvement Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Performance improvement rates plot saved to: {save_path}")

def create_performance_table(
    dataset_percentages: List[float],
    model_performance: Dict[str, Dict[str, List[float]]],
    save_path: str = './results/performance_table.csv'
):
    """
    Create a CSV table with all performance metrics.
    """
    # Create a comprehensive dataframe
    rows = []
    
    for model_name, metrics in model_performance.items():
        for i, pct in enumerate(dataset_percentages):
            rows.append({
                'Model': model_name.replace('_', ' ').title(),
                'Dataset_Size_Percent': pct,
                'F1_Score': metrics['f1_score'][i],
                'Accuracy': metrics['accuracy'][i]
            })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"Performance table saved to: {save_path}")
    return df

def plot_model_comparison_heatmap(
    dataset_percentages: List[float],
    model_performance: Dict[str, Dict[str, List[float]]],
    metric: str = 'f1_score',
    save_path: str = './results/performance_heatmap.png'
):
    """
    Create a heatmap showing model performance across different dataset sizes.
    """
    # Prepare data for heatmap
    models = list(model_performance.keys())
    data = []
    
    for model in models:
        data.append(model_performance[model][metric])
    
    # Create DataFrame
    df = pd.DataFrame(data, 
                     index=[m.replace('_', ' ').title() for m in models], 
                     columns=[f'{p}%' for p in dataset_percentages])
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': metric.replace('_', ' ').title()})
    plt.title(f'{metric.replace("_", " ").title()} Across Dataset Sizes', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Dataset Size', fontsize=12, fontweight='bold')
    plt.ylabel('Models', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Performance heatmap saved to: {save_path}")

def example_usage():
    """
    Example of how to use the plotting functions with sample data.
    """
    print("="*60)
    print("DATASET SIZE SCALING VISUALIZATION - EXAMPLE")
    print("="*60)
    
    # Example data - replace with your actual data
    dataset_percentages = [25, 50, 75, 100]
    
    # Example performance data for different models
    model_performance = {
        'logistic_regression': {
            'f1_score': [ 0.658, 0.672, 0.696, 0.701],
            'accuracy': [0.72, 0.726, 0.747, 0.756]
        },
        'transformer_model': {
            'f1_score': [0.701, 0.757, 0.771, 0.763],
            'accuracy': [0.775, 0.808, 0.818, 0.821]
        },
        'transformer_model_lem': {
            'f1_score': [0.689, 0.764, 0.745, 0.765],
            'accuracy': [0.764, 0.798, 0.791, 0.814]
        }
    }
    
    # Create all visualizations
    plot_size_scaling(dataset_percentages, model_performance)
    plot_improvement_rates(dataset_percentages, model_performance)
    plot_model_comparison_heatmap(dataset_percentages, model_performance, 'f1_score')
    plot_model_comparison_heatmap(dataset_percentages, model_performance, 'accuracy')
    create_performance_table(dataset_percentages, model_performance)
    
    print("\n✓ All visualizations completed!")

def main():
    """
    Main function to create dataset size scaling plots.
    Replace the example data with your actual experimental results.
    """
    print("="*60)
    print("DATASET SIZE SCALING VISUALIZATION")
    print("="*60)
    print()
    print("This script creates visualizations showing how model performance")
    print("changes with different dataset sizes.")
    print()
    print("To use this script:")
    print("1. Replace the example data in the example_usage() function")
    print("2. Provide your actual dataset percentages and performance metrics")
    print("3. Run the script to generate all visualizations")
    print()
    
    # Ask user if they want to run the example
    response = input("Run with example data? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        example_usage()
    else:
        print("Please modify the script with your actual data and run again.")
        print()
        print("Expected data format:")
        print("dataset_percentages = [10, 25, 50, 75, 100]")
        print("model_performance = {")
        print("    'model_name': {")
        print("        'f1_score': [list of f1 scores for each percentage],")
        print("        'accuracy': [list of accuracy scores for each percentage]")
        print("    }")
        print("}")

if __name__ == "__main__":
    main()
