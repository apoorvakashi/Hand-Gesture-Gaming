import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Directory containing latency CSV files
TYPE = "PretrainedTF"
DIRECTORY = "latency_logs/" + TYPE + "/"
BOX_PLOT_SAVE_PATH = "results/" + TYPE + "/boxplot_latency.png"
LINE_PLOT_SAVE_PATH = "results/" + TYPE + "/lineplot_latency.png"

def load_csv_files(directory):
    """Load all CSV files in the specified directory."""
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    data_frames = []
    for file in files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        df['file'] = file  # Add a column to identify the source file
        data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True)

def calculate_metrics(data):
    """Calculate metrics for each latency type."""
    metrics = data[['capture_time', 'processing_time', 'render_time', 'total_time']].agg(['mean', 'median', 'min', 'max', 'std']).T
    metrics.columns = ['Average', 'Median', 'Min', 'Max', 'Std Dev']
    return metrics

def plot_boxplot(data, save_path=None):
    """Generate boxplots for each latency type."""
    sns.boxplot(data=data[['capture_time', 'processing_time', 'render_time', 'total_time']])
    plt.title('Latency Distribution')
    plt.xlabel('Latency Type')
    plt.ylabel('Time (ms)')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_line_trends(data, save_path=None):
    """Generate line graphs for latency trends."""
    grouped = data.groupby('file').mean().reset_index()
    plt.figure(figsize=(10, 6))
    for column in ['capture_time', 'processing_time', 'render_time', 'total_time']:
        plt.plot(grouped['file'], grouped[column], marker='o', label=column)
    plt.title('Latency Trends Across Runs')
    plt.xlabel('Run')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    # Load data
    data = load_csv_files(DIRECTORY)
    
    # Calculate and display metrics
    metrics = calculate_metrics(data)
    print("Latency Metrics:")
    print(metrics)
    
    # Plot graphs
    plot_boxplot(data, save_path=BOX_PLOT_SAVE_PATH)
    plot_line_trends(data, save_path=LINE_PLOT_SAVE_PATH)

if __name__ == "__main__":
    main()
