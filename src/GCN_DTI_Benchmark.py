import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    median_absolute_error,
    explained_variance_score
)
from scipy.stats import pearsonr

# Load data
dataset = sys.argv[1]
splitmode = sys.argv[2]
modelname = sys.argv[3]
epochs=int(sys.argv[4])
data_dir = f"/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/BindDBdata/{dataset}_{splitmode}/"
train_data = pd.read_parquet(data_dir + "train.parquet")
val_data = pd.read_parquet(data_dir + "val.parquet")
test_data = pd.read_parquet(data_dir + "test.parquet")
result_csv = data_dir + f"{modelname}_metrics.csv"
loss_csv = data_dir + f"{modelname}_loss.csv"
save_model = data_dir + f"{modelname}.pt"

# Initialize DataFrame to store metrics (adding 'Loss' column)
metrics_df = pd.DataFrame(columns=['Epoch', 'Dataset', 'Loss', 'RMSE', 'MAE', 'MSE', 'R2'])

# Define dataset
class DTI_GraphDataset:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def create_edge_index(self, x, k=5):
        from torch_geometric.nn import knn_graph
        return knn_graph(x.squeeze(1), k=k)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        x = torch.tensor(np.concatenate([row['Drug_Features'], row['Target_Features']]), dtype=torch.float)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        edge_index = self.create_edge_index(x, k=5)
        label = torch.tensor([row['Affinity']], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, y=label)

# Define Graph Neural Network Model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNModel, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index, batch):
        x = self.activation(self.gcn1(x, edge_index))
        x = self.activation(self.gcn2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x.squeeze()

# Prepare dataset and loaders
SEED = 42

train_dataset = DTI_GraphDataset(train_data)
val_dataset = DTI_GraphDataset(val_data)
test_dataset = DTI_GraphDataset(test_data)

train_loader = GeoDataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = GeoDataLoader(val_dataset, batch_size=8)
test_loader = GeoDataLoader(test_dataset, batch_size=8)

# Initialize model, optimizer, and loss function
input_dim = len(train_data.iloc[0]['Drug_Features']) + len(train_data.iloc[0]['Target_Features'])
hidden_dim = 64
model = GNNModel(input_dim=input_dim, hidden_dim=hidden_dim)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# Training and Validation Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_predictions, train_labels = [], []

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.batch).view(-1)
        loss = criterion(outputs, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        train_predictions.extend(outputs.detach().cpu().numpy())
        train_labels.extend(batch.y.cpu().numpy())

    # Average train loss
    train_epoch_loss = total_loss / len(train_loader)

    # Calculate train metrics
    train_rmse = np.sqrt(mean_squared_error(train_labels, train_predictions))
    train_mae = mean_absolute_error(train_labels, train_predictions)
    train_mse = mean_squared_error(train_labels, train_predictions)
    train_r2 = r2_score(train_labels, train_predictions)

    # Validation metrics
    model.eval()
    val_total_loss = 0
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch.x, batch.edge_index, batch.batch)
            # Ensure outputs and batch.y are flattened for consistent handling
            outputs = outputs.view(-1)
            val_predictions.extend(outputs.cpu().numpy())
            val_labels.extend(batch.y.view(-1).cpu().numpy())
            val_loss = criterion(outputs, batch.y.view(-1))
            val_total_loss += val_loss.item()

    # Average val loss
    val_epoch_loss = val_total_loss / len(val_loader)

    # Calculate validation metrics
    val_rmse = np.sqrt(mean_squared_error(val_labels, val_predictions))
    val_mae = mean_absolute_error(val_labels, val_predictions)
    val_mse = mean_squared_error(val_labels, val_predictions)
    val_r2 = r2_score(val_labels, val_predictions)

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_epoch_loss:.4f} | Train RMSE: {train_rmse:.4f}, "
            f"Val Loss: {val_epoch_loss:.4f} | Val RMSE: {val_rmse:.4f}"
        )

    # Store training metrics in DataFrame
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([{
            'Epoch': epoch + 1,
            'Dataset': 'Train',
            'Loss': train_epoch_loss,
            'RMSE': train_rmse,
            'MAE': train_mae,
            'MSE': train_mse,
            'R2': train_r2
        }])],
        ignore_index=True
    )

    # Store validation metrics in DataFrame
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([{
            'Epoch': epoch + 1,
            'Dataset': 'Validation',
            'Loss': val_epoch_loss,
            'RMSE': val_rmse,
            'MAE': val_mae,
            'MSE': val_mse,
            'R2': val_r2
        }])],
        ignore_index=True
    )


# Helper function to evaluate a dataset
def evaluate_dataset(model, data_loader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in data_loader:

            outputs = model(batch.x, batch.edge_index, batch.batch).view(-1)
            predictions.extend(outputs.cpu().numpy())
            labels.extend(batch.y.cpu().numpy())

    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Compute metrics
    rmse = np.sqrt(mean_squared_error(labels, predictions))
    mae = mean_absolute_error(labels, predictions)
    mse = mean_squared_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    pearson_corr, _ = pearsonr(labels, predictions)   # correlation & p-value
    median_ae = median_absolute_error(labels, predictions)
    explained_variance = explained_variance_score(labels, predictions)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'Pearson': pearson_corr,
        'Median_AE': median_ae,
        'Explained_Variance': explained_variance
    }

# 1) Prepare a DataFrame with a predefined schema (columns + dtypes)
final_eval_df = pd.DataFrame({
    'Model': pd.Series([], dtype='object'),
    'Dataset': pd.Series([], dtype='object'),
    'RMSE': pd.Series([], dtype='float'),
    'MAE': pd.Series([], dtype='float'),
    'MSE': pd.Series([], dtype='float'),
    'R2': pd.Series([], dtype='float'),
    'Pearson': pd.Series([], dtype='float'),
    'Median_AE': pd.Series([], dtype='float'),
    'Explained_Variance': pd.Series([], dtype='float')
})

# 2) Evaluate Training Set (one by one; no loop)
train_metrics = evaluate_dataset(model, train_loader)
print(
    f"Final Training Metrics for {modelname}:\n"
    f"  RMSE: {train_metrics['RMSE']:.4f}\n"
    f"  MAE: {train_metrics['MAE']:.4f}\n"
    f"  MSE: {train_metrics['MSE']:.4f}\n"
    f"  R2: {train_metrics['R2']:.4f}\n"
    f"  Pearson: {train_metrics['Pearson']:.4f}\n"
    f"  Median_AE: {train_metrics['Median_AE']:.4f}\n"
    f"  Explained_Variance: {train_metrics['Explained_Variance']:.4f}\n"
)

train_row_dict = {
    'Model': modelname,
    'Dataset': 'Training',
    'RMSE': train_metrics['RMSE'],
    'MAE': train_metrics['MAE'],
    'MSE': train_metrics['MSE'],
    'R2': train_metrics['R2'],
    'Pearson': train_metrics['Pearson'],
    'Median_AE': train_metrics['Median_AE'],
    'Explained_Variance': train_metrics['Explained_Variance']
}

# Use .loc[len(final_eval_df)] to create a new row
final_eval_df.loc[len(final_eval_df)] = train_row_dict

# 3) Evaluate Validation Set
val_metrics = evaluate_dataset(model, val_loader)
print(
    f"Final Validation Metrics for {modelname}:\n"
    f"  RMSE: {val_metrics['RMSE']:.4f}\n"
    f"  MAE: {val_metrics['MAE']:.4f}\n"
    f"  MSE: {val_metrics['MSE']:.4f}\n"
    f"  R2: {val_metrics['R2']:.4f}\n"
    f"  Pearson: {val_metrics['Pearson']:.4f}\n"
    f"  Median_AE: {val_metrics['Median_AE']:.4f}\n"
    f"  Explained_Variance: {val_metrics['Explained_Variance']:.4f}\n"
)

val_row_dict = {
    'Model': modelname,
    'Dataset': 'Validation',
    'RMSE': val_metrics['RMSE'],
    'MAE': val_metrics['MAE'],
    'MSE': val_metrics['MSE'],
    'R2': val_metrics['R2'],
    'Pearson': val_metrics['Pearson'],
    'Median_AE': val_metrics['Median_AE'],
    'Explained_Variance': val_metrics['Explained_Variance']
}

final_eval_df.loc[len(final_eval_df)] = val_row_dict

# 4) Evaluate Test Set
test_metrics = evaluate_dataset(model, test_loader)
print(
    f"Final Test Metrics for {modelname}:\n"
    f"  RMSE: {test_metrics['RMSE']:.4f}\n"
    f"  MAE: {test_metrics['MAE']:.4f}\n"
    f"  MSE: {test_metrics['MSE']:.4f}\n"
    f"  R2: {test_metrics['R2']:.4f}\n"
    f"  Pearson: {test_metrics['Pearson']:.4f}\n"
    f"  Median_AE: {test_metrics['Median_AE']:.4f}\n"
    f"  Explained_Variance: {test_metrics['Explained_Variance']:.4f}\n"
)

test_row_dict = {
    'Model': modelname,
    'Dataset': 'Test',
    'RMSE': test_metrics['RMSE'],
    'MAE': test_metrics['MAE'],
    'MSE': test_metrics['MSE'],
    'R2': test_metrics['R2'],
    'Pearson': test_metrics['Pearson'],
    'Median_AE': test_metrics['Median_AE'],
    'Explained_Variance': test_metrics['Explained_Variance']
}

final_eval_df.loc[len(final_eval_df)] = test_row_dict

# 5) Inspect the final DataFrame
print("\nFinal Evaluation DataFrame:\n", final_eval_df)

# Save metrics to CSV
metrics_df.to_csv(loss_csv, index=False)
print('Metrics saved!')

final_eval_df.to_csv(result_csv,index=False)
print('Final results saved',result_csv)

# Save model
torch.save(model.state_dict(), save_model)
print('Model saved!',save_model)