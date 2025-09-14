import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
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
class DTI_Dataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        return {
            'Drug_Features': torch.tensor(np.asarray(row['Drug_Features'], dtype=np.float32)),
            'Target_Features': torch.tensor(np.asarray(row['Target_Features'], dtype=np.float32)),
            'Affinity': torch.tensor(float(row['Affinity']), dtype=torch.float32),
        }

# Define Deep Cross Network Model
class DeepCrossNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_cross_layers):
        super(DeepCrossNetwork, self).__init__()
        self.deep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        self.cross_layers = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(num_cross_layers)])
        self.fc = nn.Linear(input_dim + hidden_dim // 2, 1)

    def forward(self, x):
        # Cross layers
        cross_x = x
        for layer in self.cross_layers:
            cross_x = x + layer(cross_x)

        # Deep layers
        deep_x = self.deep(x)

        # Concatenate cross and deep layers
        combined_x = torch.cat((cross_x, deep_x), dim=-1)
        output = self.fc(combined_x)
        return output.squeeze()

# Example dataframes: train_data, val_data, test_data
# Split into train, validation, and test sets
SEED = 42
train_dataset = DTI_Dataset(train_data)
val_dataset = DTI_Dataset(val_data)
test_dataset = DTI_Dataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Verify feature dimensions
for batch in train_loader:
    drug_input_dim = batch['Drug_Features'].shape[1]
    protein_input_dim = batch['Target_Features'].shape[1]
    break

# Initialize model, optimizer, and loss function
input_dim = drug_input_dim + protein_input_dim
hidden_dim = 512
num_cross_layers = 3
model = DeepCrossNetwork(input_dim=input_dim, hidden_dim=hidden_dim, num_cross_layers=num_cross_layers)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

# ----- Training and Validation -----

for epoch in range(epochs):
    model.train()
    total_loss = 0
    train_predictions, train_labels = [], []

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        drug_features = batch['Drug_Features']
        protein_features = batch['Target_Features']
        labels = batch['Affinity']

        # Concatenate features
        features = torch.cat((drug_features, protein_features), dim=-1)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        train_predictions.extend(outputs.detach().cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    # Average train loss
    train_epoch_loss = total_loss / len(train_loader)

    # Calculate train metrics
    train_rmse = np.sqrt(mean_squared_error(train_labels, train_predictions))
    train_mae = mean_absolute_error(train_labels, train_predictions)
    train_mse = mean_squared_error(train_labels, train_predictions)
    train_r2 = r2_score(train_labels, train_predictions)

    # Validation pass
    model.eval()
    val_total_loss = 0
    val_predictions, val_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            drug_features = batch['Drug_Features']
            protein_features = batch['Target_Features']
            labels = batch['Affinity']

            features = torch.cat((drug_features, protein_features), dim=-1)
            outputs = model(features)
            val_loss = criterion(outputs, labels)
            val_total_loss += val_loss.item()

            val_predictions.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

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

# ----- Final Evaluation (Train, Validation, Test) -----

# Helper function to evaluate a dataset
def evaluate_dataset(model, data_loader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            drug_features = batch['Drug_Features']
            protein_features = batch['Target_Features']
            y = batch['Affinity']

            # Concatenate
            features = torch.cat((drug_features, protein_features), dim=-1)
            out = model(features).cpu().numpy()
            predictions.extend(out)
            labels.extend(y.cpu().numpy())

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
print('Final results saved',final_eval_df)

# Save model
torch.save(model.state_dict(), save_model)
print('Model saved!',save_model)