#!/usr/bin/env python3
"""Parameter sweep for photon deconvolution model"""
import tensorflow as tf
import numpy as np
from itertools import product
import pandas as pd

# Parameter grid
param_grid = {
    'lr': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
    'batch_size': [32, 64, 128],
    'filters': [[16, 8], [12, 6], [20, 10], [24, 12]],
    'kernels': [[9, 7, 5], [7, 5, 3], [11, 9, 7]]
}

def load_data():
    """Load and split dataset"""
    data = np.load("../output/ml_training_dataset.npz", mmap_mode='r')
    waveforms, photon_times, time_axis = data['waveforms'], data['photon_times_detected_jittered'], data['time_axis']
    
    N = len(waveforms)
    idx = np.random.RandomState(42).permutation(N)
    splits = [int(0.7*N), int(0.9*N)]
    train_idx, val_idx, test_idx = idx[:splits[0]], idx[splits[0]:splits[1]], idx[splits[1]:]
    
    return waveforms, photon_times, time_axis, (train_idx, val_idx, test_idx)

def create_targets(photon_times, time_axis):
    """Convert photon times to time-aligned counts"""
    targets = np.zeros((len(photon_times), len(time_axis)), dtype=np.float32)
    for i, times in enumerate(photon_times):
        valid_times = times[~np.isnan(times)]
        if len(valid_times) > 0:
            # Map each photon time to nearest time_axis index
            indices = np.searchsorted(time_axis, valid_times)
            indices = np.clip(indices, 0, len(time_axis) - 1)
            # Count photons at each time point
            unique_indices, counts = np.unique(indices, return_counts=True)
            targets[i, unique_indices] = counts.astype(np.float32)
    return targets

def build_model(input_len, filters, kernels, lr):
    """Build 3-layer CNN"""
    x = inputs = tf.keras.Input(shape=(input_len,))
    x = tf.keras.layers.Reshape((input_len, 1))(x)
    x = tf.keras.layers.Conv1D(filters[0], kernels[0], activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1D(filters[1], kernels[1], activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv1D(1, kernels[2], activation='relu', padding='same')(x)
    outputs = tf.keras.layers.Reshape((input_len,))(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='poisson', metrics=['mae'])
    return model

def run_experiment(params, X_train, y_train, X_val, y_val, X_test, y_test):
    """Run single experiment"""
    model = build_model(X_train.shape[1], params['filters'], params['kernels'], params['lr'])
    
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7, verbose=0),
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, verbose=0)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                       batch_size=params['batch_size'], epochs=30, 
                       callbacks=callbacks, verbose=0)
    
    # Get best epoch metrics
    best_epoch = np.argmin(history.history['val_loss'])
    train_loss = history.history['loss'][best_epoch]
    val_loss = history.history['val_loss'][best_epoch]
    
    # Test correlation
    y_pred = model.predict(X_test, verbose=0)
    pred_counts = np.sum(y_pred, axis=1)
    true_counts = np.sum(y_test, axis=1)
    correlation = np.corrcoef(pred_counts, true_counts)[0, 1]
    
    return {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'correlation': correlation,
        'n_params': model.count_params()
    }

if __name__ == "__main__":
    import os
    
    # Load data
    waveforms, photon_times, time_axis, (train_idx, val_idx, test_idx) = load_data()
    targets = create_targets(photon_times, time_axis)
    
    X_train, y_train = waveforms[train_idx], targets[train_idx]
    X_val, y_val = waveforms[val_idx], targets[val_idx]
    X_test, y_test = waveforms[test_idx], targets[test_idx]
    
    print(f"Data loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    # Get all parameter combinations
    all_combinations = list(product(*param_grid.values()))
    total_runs = len(all_combinations)
    
    # Check if running as SLURM job array
    job_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if job_id is not None:
        job_id = int(job_id)
        total_jobs = 100  # Fixed number of jobs
        
        # Split combinations across jobs
        combinations_per_job = total_runs // total_jobs
        remainder = total_runs % total_jobs
        
        start_idx = job_id * combinations_per_job + min(job_id, remainder)
        if job_id < remainder:
            end_idx = start_idx + combinations_per_job + 1
        else:
            end_idx = start_idx + combinations_per_job
            
        job_combinations = all_combinations[start_idx:end_idx]
        print(f"Job {job_id}/{total_jobs}: Running {len(job_combinations)} combinations ({start_idx}-{end_idx-1})")
        output_file = f'param_sweep_results_job_{job_id:02d}.csv'
    else:
        # Run all combinations (original behavior)
        job_combinations = all_combinations
        print(f"Running all {total_runs} combinations")
        output_file = 'param_sweep_results.csv'
    
    # Run parameter sweep
    results = []
    
    for i, (lr, batch_size, filters, kernels) in enumerate(job_combinations):
        params = {'lr': lr, 'batch_size': batch_size, 'filters': filters, 'kernels': kernels}
        
        print(f"Run {i+1}/{len(job_combinations)}: lr={lr}, bs={batch_size}, filters={filters}, kernels={kernels}")
        
        try:
            result = run_experiment(params, X_train, y_train, X_val, y_val, X_test, y_test)
            result.update(params)
            results.append(result)
            print(f"  -> Loss: {result['val_loss']:.4f}, Corr: {result['correlation']:.3f}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values('val_loss')
        df.to_csv(output_file, index=False)
        
        print(f"\nParameter sweep complete! Results saved to {output_file}")
        print(f"Completed {len(results)} successful runs out of {len(job_combinations)} attempted")
        if len(results) > 0:
            print("\nBest configurations for this job:")
            print(df.head()[['lr', 'batch_size', 'filters', 'kernels', 'val_loss', 'correlation']].to_string())
    else:
        print(f"\nNo successful runs completed")