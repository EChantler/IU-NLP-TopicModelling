import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_data_from_mlflow():
    """Load data from MLflow artifacts following the pattern from training scripts"""
    print("Loading data from MLflow artifacts...")
    
    client = MlflowClient()
    try:
        # Get all experiments and search for cleaned_data artifacts
        experiments = client.search_experiments()
        data_run = None
        
        for experiment in experiments:
            runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])
            for run in runs:
                try:
                    artifacts = client.list_artifacts(run.info.run_id, path="cleaned_data")
                    if artifacts:  # Found a run with cleaned_data
                        data_run = run
                        print(f"Found cleaned_data in experiment {experiment.name} (ID: {experiment.experiment_id})")
                        break
                except Exception:
                    continue
            if data_run:
                break
        
        if data_run:
            print(f"Loading data from MLflow run: {data_run.info.run_id}")
            # Download data artifacts
            data_path = mlflow.artifacts.download_artifacts(artifact_path="cleaned_data", run_id=data_run.info.run_id)
            
            # Load all datasets
            datasets = {}
            
            # Test datasets
            datasets['test_tfidf'] = pd.read_csv(os.path.join(data_path, 'test_tfidf.csv'))
            datasets['test_transformer'] = pd.read_csv(os.path.join(data_path, 'test_transformer.csv'))
            datasets['test_transformer_lem'] = pd.read_csv(os.path.join(data_path, 'test_transformer_lem.csv'))
            
            # Training datasets (for domain analysis)
            datasets['train_tfidf'] = pd.read_csv(os.path.join(data_path, 'train_tfidf.csv'))
            datasets['train_transformer'] = pd.read_csv(os.path.join(data_path, 'train_transformer.csv'))
            datasets['train_transformer_lem'] = pd.read_csv(os.path.join(data_path, 'train_transformer_lem.csv'))
            
            print(f"Loaded data from MLflow artifacts:")
            for name, df in datasets.items():
                print(f"  {name}: {len(df)} samples")
            
            return datasets
            
        else:
            raise FileNotFoundError("No runs with cleaned_data artifacts found in any experiment")
            
    except Exception as e:
        print(f"Failed to load data from MLflow: {e}")
        raise RuntimeError(f"Failed to load data from registry: {e}")
    
def load_logistic_regression_model():
    """Load the trained logistic regression model"""
    try:
        model = joblib.load('./results/target_pipe.joblib')
        print("✓ Loaded logistic regression model from local file")
        return model
    except FileNotFoundError:
        print("✗ Could not find local logistic regression model")
        
        # Try to load from MLflow
        try:
            client = MlflowClient()
            model_name = "logistic_regression_target_predictor"
            
            # Get the latest version of the model
            latest_versions = client.get_latest_versions(model_name, stages=["None"])
            if latest_versions:
                model_version = latest_versions[0].version
                model_uri = f"models:/{model_name}/{model_version}"
                model = mlflow.sklearn.load_model(model_uri)
                print(f"✓ Loaded logistic regression model from MLflow (version {model_version})")
                return model
            else:
                raise ValueError(f"No versions found for model {model_name}")
        except Exception as e:
            print(f"✗ Failed to load logistic regression model from MLflow: {e}")
            return None

def load_transformer_model(model_path, model_name_mlflow):
    """Load a transformer model from local path or MLflow"""
    try:
        # Try loading from local path first
        if os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            print(f"✓ Loaded {model_name_mlflow} from local path: {model_path}")
            return model, tokenizer
        else:
            print(f"✗ Local path not found: {model_path}")
    except Exception as e:
        print(f"✗ Failed to load from local path: {e}")
    
    # Try loading from MLflow
    try:
        client = MlflowClient()
        latest_versions = client.get_latest_versions(model_name_mlflow, stages=["None"])
        if latest_versions:
            model_version = latest_versions[0].version
            model_uri = f"models:/{model_name_mlflow}/{model_version}"
            
            # Load the PyTorch model
            model = mlflow.pytorch.load_model(model_uri)
            
            # For tokenizer, we need to load from the original model name
            tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            
            print(f"✓ Loaded {model_name_mlflow} from MLflow (version {model_version})")
            return model, tokenizer
        else:
            raise ValueError(f"No versions found for model {model_name_mlflow}")
    except Exception as e:
        print(f"✗ Failed to load {model_name_mlflow} from MLflow: {e}")
        return None, None

def predict_transformer(model, tokenizer, texts, batch_size=8):
    """Make predictions using transformer model"""
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_preds)
    
    return np.array(predictions)

def plot_confusion_matrix(y_true, y_pred, all_labels, title, save_path):
    """Plot and save confusion matrix"""
    # Use all possible labels to ensure consistent matrix size across evaluations
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_labels, yticklabels=all_labels)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm

def evaluate_model_on_data(model, data, model_type, dataset_name, target_labels):
    """Evaluate a model on given data and return metrics"""
    print(f"\n=== Evaluating {model_type} on {dataset_name} ===")
    
    if model_type == 'logistic_regression':
        predictions = model.predict(data['text'])
    elif model_type in ['transformer', 'transformer_lemmatized']:
        # Use raw text without domain prefix (models should be retrained without domain)
        input_texts = data['text']
        predictions = predict_transformer(model[0], model[1], input_texts.tolist())
        
        # Convert numeric predictions back to string labels
        # Get label mapping from model
        if hasattr(model[0], 'config') and hasattr(model[0].config, 'id2label'):
            id2label = model[0].config.id2label
            print(f"Model's id2label mapping: {id2label}")
            predictions = [id2label[pred] for pred in predictions]
        else:
            # Fallback: assume predictions are already in the correct format or need simple mapping
            print(f"Warning: No id2label mapping found. Predictions may need manual mapping.")
            print(f"Prediction sample: {predictions[:5] if len(predictions) > 0 else 'None'}")
            print(f"Expected target labels: {target_labels}")
    
    # Use only target labels that are actually present in the current test data
    actual_target_labels = sorted(set(data['target']))
    print(f"Unique predictions made by {model_type}: {sorted(set(predictions))}")
    print(f"Target labels present in data: {actual_target_labels}")
    print(f"Evaluating on present target labels only (excluding zero topics)")
    
    # Calculate metrics using only the target labels present in this dataset
    accuracy = accuracy_score(data['target'], predictions)
    f1_macro = f1_score(data['target'], predictions, labels=actual_target_labels, average='macro', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-macro: {f1_macro:.4f}")
    
    # Generate classification report - use only present target labels
    report = classification_report(data['target'], predictions, labels=actual_target_labels, digits=4, zero_division=0)
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix with only present target labels for consistency
    os.makedirs('./results/evaluation', exist_ok=True)
    cm_path = f'./results/evaluation/{model_type}_{dataset_name}_confusion_matrix.png'
    cm = plot_confusion_matrix(data['target'], predictions, actual_target_labels, 
                              f'{model_type.title()} - {dataset_name.title()}', cm_path)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'predictions': predictions,
        'classification_report': report,
        'confusion_matrix': cm
    }

def evaluate_by_domain(model, data, model_type, target_labels):
    """Evaluate model performance by domain"""
    print(f"\n=== Domain Analysis for {model_type} ===")
    
    domains = data['domain'].unique()
    domain_results = {}
    
    for domain in domains:
        domain_data = data[data['domain'] == domain].copy()
        print(f"\nDomain: {domain} ({len(domain_data)} samples)")
        
        # Pass the original target_labels parameter but the function will use only present labels
        result = evaluate_model_on_data(model, domain_data, model_type, 
                                      f'{domain}_domain', target_labels)
        domain_results[domain] = result
    
    return domain_results

def save_results_summary(results, filename):
    """Save evaluation results summary to file"""
    with open(filename, 'w') as f:
        f.write("MODEL EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        for model_name, model_results in results.items():
            f.write(f"{model_name.upper()}\n")
            f.write("-" * len(model_name) + "\n")
            
            # Test set results
            f.write("Test Set Performance:\n")
            f.write(f"  Accuracy: {model_results['test']['accuracy']:.4f}\n")
            f.write(f"  F1-macro: {model_results['test']['f1_macro']:.4f}\n\n")
            
            # Domain results
            f.write("Performance by Domain (Training Set):\n")
            for domain, domain_result in model_results['domains'].items():
                f.write(f"  {domain}:\n")
                f.write(f"    Accuracy: {domain_result['accuracy']:.4f}\n")
                f.write(f"    F1-macro: {domain_result['f1_macro']:.4f}\n")
            f.write("\n")
        
        f.write("\nDetailed classification reports are saved separately for each model and domain.\n")

def create_performance_comparison_plot(results):
    """Create comparison plots for model performance"""
    # Prepare data for plotting
    models = list(results.keys())
    test_accuracies = [results[model]['test']['accuracy'] for model in models]
    test_f1_scores = [results[model]['test']['f1_macro'] for model in models]
    
    # Domain performance data
    domains = ['academic', 'social', 'news']
    domain_data = {domain: [] for domain in domains}
    
    for model in models:
        for domain in domains:
            if domain in results[model]['domains']:
                domain_data[domain].append(results[model]['domains'][domain]['f1_macro'])
            else:
                domain_data[domain].append(0)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Test set performance comparison
    x_pos = np.arange(len(models))
    
    # Accuracy comparison
    axes[0, 0].bar(x_pos, test_accuracies, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 0].set_title('Test Set Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(test_accuracies):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # F1-macro comparison
    axes[0, 1].bar(x_pos, test_f1_scores, color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[0, 1].set_title('Test Set F1-Macro Comparison', fontweight='bold')
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('F1-Macro Score')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].set_ylim(0, 1)
    
    # Add value labels on bars
    for i, v in enumerate(test_f1_scores):
        axes[0, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # Domain performance comparison
    x_pos_domain = np.arange(len(domains))
    width = 0.25
    
    for i, model in enumerate(models):
        model_domain_scores = [domain_data[domain][i] for domain in domains]
        axes[1, 0].bar(x_pos_domain + i*width, model_domain_scores, width, 
                       label=model, alpha=0.8)
    
    axes[1, 0].set_title('F1-Macro by Domain (Training Set)', fontweight='bold')
    axes[1, 0].set_xlabel('Domains')
    axes[1, 0].set_ylabel('F1-Macro Score')
    axes[1, 0].set_xticks(x_pos_domain + width)
    axes[1, 0].set_xticklabels(domains)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1)
    
    # Model performance heatmap
    heatmap_data = []
    for model in models:
        row = [results[model]['test']['f1_macro']]
        for domain in domains:
            if domain in results[model]['domains']:
                row.append(results[model]['domains'][domain]['f1_macro'])
            else:
                row.append(0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, 
                             index=models, 
                             columns=['Test Set'] + domains)
    
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='YlOrRd', 
                ax=axes[1, 1], cbar_kws={'label': 'F1-Macro Score'})
    axes[1, 1].set_title('Performance Heatmap', fontweight='bold')
    axes[1, 1].set_xlabel('Dataset/Domain')
    axes[1, 1].set_ylabel('Models')
    
    plt.tight_layout()
    plt.savefig('./results/evaluation/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main evaluation function"""
    print("="*60)
    print("MODEL EVALUATION SCRIPT")
    print("="*60)
    
    # Create results directory
    os.makedirs('./results/evaluation', exist_ok=True)
    
    # Load data from MLflow artifacts
    datasets = load_data_from_mlflow()
    
    # Extract individual datasets
    test_tfidf = datasets['test_tfidf']
    test_transformer = datasets['test_transformer']
    test_transformer_lem = datasets['test_transformer_lem']
    train_tfidf = datasets['train_tfidf']
    train_transformer = datasets['train_transformer']
    train_transformer_lem = datasets['train_transformer_lem']
    
    # Get target labels (consistent across all datasets)
    target_labels = sorted(test_tfidf['target'].unique())
    print(f"\nTarget labels: {target_labels}")
    
    # Log dataset information
    print(f"\nDataset information:")
    print(f"  Target classes: {test_tfidf['target'].nunique()}")
    print(f"  Domains: {test_tfidf['domain'].nunique()}")
    print(f"  Domain distribution in test set: {test_tfidf['domain'].value_counts().to_dict()}")
    
    # Load models
    print("\n" + "="*60)
    print("LOADING MODELS")
    print("="*60)
    
    lr_model = load_logistic_regression_model()
    transformer_model, transformer_tokenizer = load_transformer_model(
        './results/final_transformer_model', 'transformer_model')
    transformer_lem_model, transformer_lem_tokenizer = load_transformer_model(
        './results/final_transformer_model_lemmatized', 'transformer_model_lemmatized')
    
    # Store results
    results = {}
    
    # Evaluate Logistic Regression
    if lr_model is not None:
        print("\n" + "="*60)
        print("LOGISTIC REGRESSION EVALUATION")
        print("="*60)
        
        # Test set evaluation
        lr_test_results = evaluate_model_on_data(
            lr_model, test_tfidf, 'logistic_regression', 'test_set', target_labels)
        
        # Domain evaluation on training set
        lr_domain_results = evaluate_by_domain(
            lr_model, train_tfidf, 'logistic_regression', target_labels)
        
        results['logistic_regression'] = {
            'test': lr_test_results,
            'domains': lr_domain_results
        }
    
    # Evaluate Transformer Model
    if transformer_model is not None and transformer_tokenizer is not None:
        print("\n" + "="*60)
        print("TRANSFORMER MODEL EVALUATION")
        print("="*60)
        
        # Test set evaluation
        transformer_test_results = evaluate_model_on_data(
            (transformer_model, transformer_tokenizer), test_transformer, 
            'transformer', 'test_set', target_labels)
        
        # Domain evaluation on training set
        transformer_domain_results = evaluate_by_domain(
            (transformer_model, transformer_tokenizer), train_transformer, 
            'transformer', target_labels)
        
        results['transformer'] = {
            'test': transformer_test_results,
            'domains': transformer_domain_results
        }
    
    # Evaluate Transformer Lemmatized Model
    if transformer_lem_model is not None and transformer_lem_tokenizer is not None:
        print("\n" + "="*60)
        print("TRANSFORMER LEMMATIZED MODEL EVALUATION")
        print("="*60)
        
        # Test set evaluation
        transformer_lem_test_results = evaluate_model_on_data(
            (transformer_lem_model, transformer_lem_tokenizer), test_transformer_lem, 
            'transformer_lemmatized', 'test_set', target_labels)
        
        # Domain evaluation on training set
        transformer_lem_domain_results = evaluate_by_domain(
            (transformer_lem_model, transformer_lem_tokenizer), train_transformer_lem, 
            'transformer_lemmatized', target_labels)
        
        results['transformer_lemmatized'] = {
            'test': transformer_lem_test_results,
            'domains': transformer_lem_domain_results
        }
    
    # Save results summary
    if results:
        print("\n" + "="*60)
        print("SAVING RESULTS AND GENERATING COMPARISONS")
        print("="*60)
        
        save_results_summary(results, './results/evaluation/evaluation_summary.txt')
        create_performance_comparison_plot(results)
        
        print("\n✓ Evaluation complete!")
        print("✓ Results saved to ./results/evaluation/")
        print("✓ Confusion matrices saved as PNG files")
        print("✓ Summary saved to evaluation_summary.txt")
        print("✓ Comparison plots saved to model_comparison.png")
    else:
        print("\n✗ No models could be loaded for evaluation")

if __name__ == "__main__":
    main()
