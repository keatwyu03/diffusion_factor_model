"""
Training script for Diffusion Factor Model
"""

import torch
import numpy as np
from torch.utils.data import TensorDataset
import os
import gc
import argparse
import time

from diffusion_factor_model.diffusion_factor_model import Unet, GaussianDiffusion, Trainer
import config.config as config
from h_function import HFunctionTrainer

def find_latest_checkpoint():
    """Return the most recently modified model-epoch-*.pt file under MODELS_DIR."""
    import glob
    pattern = os.path.join(config.MODELS_DIR, "**", "model-epoch-*.pt")
    ckpts = glob.glob(pattern, recursive=True)
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)

def get_dim_mults_for_size(height, width):
    """
    Determine appropriate dimension multipliers for UNet based on input dimensions.
    The dimensions must be divisible by the maximum downsampling factor.
    
    Args:
        height: Height of input
        width: Width of input
        
    Returns:
        Tuple of dimension multipliers suitable for the input size
    """
    # Calculate the maximum downsampling factor possible
    min_dim = min(height, width)
    
    if min_dim >= 32:
        return config.DIM_MULTS_LARGE  # Standard for large inputs
    elif min_dim >= 16:
        return config.DIM_MULTS_MEDIUM  # For medium inputs
    elif min_dim >= 8:
        return config.DIM_MULTS_SMALL   # For small inputs
    elif min_dim >= 4:
        return config.DIM_MULTS_TINY    # For very small inputs
    else:
        return config.DIM_MULTS_MINIMAL # Minimal case

def train_model(data_path, seed=None, num_samples=None, gpu_id=0, epochs=None, save_timesteps=None, skip_hfunction=False, h_ckpt=None, skip_diffusion_training=False, diffusion_ckpt=None, test_days=None):
    """
    Train the diffusion model using a specific data file

    Args:
        data_path: Path to the data file to use for training
        seed: Random seed for reproducibility
        num_samples: Number of training samples to use (None = use all)
        gpu_id: GPU ID to use
        epochs: Number of epochs to train (None = use config.EPOCHS)
        save_timesteps: List of specific timesteps to save during sampling for early stopping evaluation
                       (None = use config.SAVE_TIMESTEPS, which defaults to None meaning save only final result)
        test_days: Number of days to hold out as test set (None = use config.TEST_DAYS)
    """
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Use config default if save_timesteps not specified
    if save_timesteps is None:
        save_timesteps = config.SAVE_TIMESTEPS
    
    # Set seed and get timestamp for experiment ID
    seed = config.set_seed(seed)
    timestamp = int(time.time())
    
    # Get filename from path for experiment ID
    filename = os.path.basename(data_path)
    data_id = os.path.splitext(filename)[0]
    
    # Create experiment ID
    exp_id = f"{config.EXP_PREFIX}_{data_id}_ts{timestamp}_seed{seed}"
    
    # Load data to determine shape and dimensions
    if data_path.endswith(".csv"):
        import pandas as pd
        df_tmp = pd.read_csv(data_path, index_col="Date")
        data_np = df_tmp[config.TICKERS].dropna().values
    else:
        data_np = np.load(data_path)
    data_shape = data_np.shape
    print(f"Loaded data with shape: {data_shape}, dtype: {data_np.dtype}")

    # Hold out test set
    if test_days is None:
        test_days = config.TEST_DAYS
    if test_days > 0:
        test_np = data_np[-test_days:]
        data_np = data_np[:-test_days]
        print(f"Holding out last {test_days} rows as test set. Training on {len(data_np)} samples.")
    else:
        test_np = None

    # Limit number of samples if specified
    if num_samples is not None and num_samples < data_np.shape[0]:
        data_np = data_np[:num_samples]
        print(f"Using {num_samples} samples from the data")
    
    # Determine data dimensions and reshape strategy
    data_shape = data_np.shape
    if len(data_shape) == 2:
        # data (samples, features) - reshape to 2D format
        samples, features = data_shape
        
        # Try to make the image as square as possible
        width = 2**(int(np.log2(features)) // 2)
        height = features // width
        
        if height * width != features:
            # If not perfectly divisible, use a simple reshape
            height, width = 1, features
        
        # Reshape data to [samples, 1, height, width]
        data = torch.from_numpy(data_np).float()
        if data.shape[1] != features:
            print(f"Warning: Data dimension ({data.shape[1]}) doesn't match expected features ({features})")
        
        data = data.reshape(-1, 1, height, width)
        print(f"Reshaped 2D data to: {data.shape} with dimensions [batch, channels, height={height}, width={width}]")
        
    elif len(data_shape) == 3:
        # data (samples, height, width) - add channel dimension
        samples, height, width = data_shape
        
        # Convert to tensor and add channel dimension
        data = torch.from_numpy(data_np).float()
        data = data.unsqueeze(1)  # Add channel dimension [samples, 1, height, width]
        print(f"Reshaped 3D data to: {data.shape} with dimensions [batch, channels, height={height}, width={width}]")
        
    else:
        raise ValueError(f"Unsupported data shape: {data_shape}, expected 2D or 3D array")
    
    # Get appropriate dimension multipliers for UNet
    dim_mults = get_dim_mults_for_size(height, width)
    print(f"Using dimension multipliers: {dim_mults} for input size ({height}, {width})")
    
    # Calculate maximum downsampling factor
    max_downsample = 2**(len(dim_mults)-1) if len(dim_mults) > 0 else 1
    print(f"Maximum downsampling factor: {max_downsample}")
    
    # Create directories for this experiment
    model_dir = os.path.join(config.MODELS_DIR, exp_id)
    sample_dir = os.path.join(config.SAMPLES_DIR, exp_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # Create dataset
    data_mean = data.mean(dim=0, keepdim=True)[0]
    data_std = data.std(dim=0, keepdim=True)[0]
    data = (data - data_mean) / data_std
    dataset = TensorDataset(data)

    # Prepare standardized test tensor for event reporting
    if test_np is not None:
        test_data = torch.from_numpy(test_np).float().reshape(-1, 1, height, width)
        test_data = (test_data - data_mean) / data_std
    else:
        test_data = None
    
    # Calculate latent dimension (total number of features)
    latent_dim = height * width
    
    # Use epochs from argument or config
    if epochs is None:
        epochs = config.EPOCHS
    
    # Initialize model with appropriate dimension multipliers
    model = Unet(
        dim=config.MODEL_DIM,
        channels=config.MODEL_CHANNELS,
        filter_size=config.MODEL_FILTER_SIZE,
        dim_mults=dim_mults  # Use appropriate multipliers for this input size
    )
    
    print("Model initialized")
    
    # Initialize diffusion process with proper image size
    diffusion = GaussianDiffusion(
        model,
        image_size=(height, width),  # Use our reshaped dimensions
        latent_dim=latent_dim,
        timesteps=config.TIMESTEPS,
        objective=config.OBJECTIVE,
        beta_schedule=config.BETA_SCHEDULE,
        auto_normalize=config.AUTO_NORMALIZE
    )
    
    print("Diffusion process initialized")
    
    # Initialize Trainer with custom epochs and optional save_timesteps for early stopping
    trainer = Trainer(
        diffusion,
        dataset,
        train_batch_size=min(config.BATCH_SIZE, len(dataset)),  # Ensure batch size doesn't exceed dataset size
        train_lr=config.LEARNING_RATE,
        train_epochs=epochs,
        adamw_weight_decay=config.WEIGHT_DECAY,
        cosine_scheduler=config.USE_COSINE_SCHEDULER,
        warm_up=config.USE_WARM_UP,
        warmup_iters=config.WARMUP_STEPS,
        T_0=config.COSINE_CYCLE_LENGTH,
        T_mult=config.T_MULT,
        eta_min=config.COSINE_LR_MIN,
        cosine_steps=config.COSINE_STEPS,
        gradient_accumulate_every=config.GRADIENT_ACCUMULATION,
        ema_decay=config.EMA_DECAY,
        split_batches=config.SPLIT_BATCHES,
        save_and_sample_every=config.SAVE_INTERVAL,
        results_folder=model_dir,
        param_path="",
        amp=config.USE_AMP,
        save_timesteps=save_timesteps,  # Pass save_timesteps for early stopping evaluation
    )
    
    print("Trainer initialized")
    
    # Train model
    if skip_diffusion_training:
        if diffusion_ckpt is None:
            diffusion_ckpt = find_latest_checkpoint()
        if diffusion_ckpt is None or not os.path.exists(diffusion_ckpt):
            raise ValueError("No checkpoint found. Train a model first or pass --diffusion_ckpt.")
        print(f"Skipping diffusion training. Loading checkpoint from: {diffusion_ckpt}")
        trainer.load(diffusion_ckpt)
    else:
        print(f"Starting training for {epochs} epochs...")
        trainer.train()

        loss_path = os.path.join(model_dir, config.LOSS_FILENAME)
        with open(loss_path, "w") as f:
            f.write("epoch,loss\n")
            for ep, loss in trainer.train_losses:
                f.write(f"{ep},{loss}\n")
        print(f"Loss curve saved to: {loss_path}")

    # ==================== H-Function Training ====================
    print("\n" + "=" * 60)
    print("H-Function Training")
    print("=" * 60)

    h_trainer = HFunctionTrainer(
        diffusion=diffusion,
        asset_dim=height * width,
        embed_dim=config.H_EMBED_DIM,
        event_asset_idx=config.H_EVENT_ASSET_IDX,
        event_threshold=config.H_EVENT_THRESHOLD,
        device=config.DEVICE,
    )

    if skip_hfunction:
        print("Skipping H-function training.")
    elif h_ckpt is not None and os.path.exists(h_ckpt):
        h_trainer.load(h_ckpt)
    else:
        h_trainer.train(
            train_data=data,
            n_epochs=config.H_EPOCHS,
            batch_size=config.H_BATCH_SIZE,
            learning_rate=config.H_LEARNING_RATE,
            weight_decay=config.H_WEIGHT_DECAY,
            scheduler_patience=config.H_SCHEDULER_PATIENCE,
            scheduler_factor=config.H_SCHEDULER_FACTOR,
            test_data=test_data,
        )
        h_trainer.save(os.path.join(model_dir, "hfunction.pt"))

    # Generate samples
    print("Generating samples...")
    sample_batches = config.SAMPLE_BATCHES
    samples_per_batch = config.SAMPLES_PER_BATCH
    
    config.set_seed(seed)  # Reset seed for reproducibility
    
    for i in range(sample_batches):
        # Pass save_timesteps parameter to sample method for early stopping evaluation
        samples = diffusion.sample(batch_size=samples_per_batch, save_timesteps=save_timesteps)
        samples = samples.view(samples.size(0), -1).cpu().numpy()
        samples = samples * data_std.view(-1).cpu().numpy() + data_mean.view(-1).cpu().numpy()
        
        sample_file = os.path.join(sample_dir, f"sample_batch{i+1}.npy")
        np.save(sample_file, samples)
        
        # Clean up to prevent memory issues
        del samples
        gc.collect()
    
    # Clean up
    del trainer, model, diffusion, data, dataset
    gc.collect()
    
    # Write model_dir to latest_run.txt so conditional_sampling.py can find it automatically
    latest_run_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_run.txt")
    with open(latest_run_path, "w") as f:
        f.write(model_dir)

    print(f"Training and sampling complete for {exp_id}")
    print(f"Models saved to: {model_dir}")
    print(f"Samples saved to: {sample_dir}")
    print(f"\nTo generate conditional samples run:")
    print(f"  python conditional_sampling.py")

    return model_dir, sample_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train diffusion factor model on specific data file")
    parser.add_argument("--data_path", type=str, required=True, 
                      help="Path to the data file for training")
    parser.add_argument("--seed", type=int, default=None, 
                      help="Random seed")
    parser.add_argument("--num_samples", type=int, default=None, 
                      help="Number of training samples (None = use all)")
    parser.add_argument("--gpu", type=int, default=0, 
                      help="GPU ID")
    parser.add_argument("--epochs", type=int, default=None, 
                      help="Number of epochs to train (None = use config value)")
    parser.add_argument("--save_timesteps", type=int, nargs='+', default=None,
                      help="Specific timesteps to save during sampling for early stopping evaluation (e.g., --save_timesteps 100 200 500)")
    parser.add_argument("--skip_hfunction", action="store_true",
                      help="Skip H-function training")
    parser.add_argument("--h_ckpt", type=str, default=None,
                      help="Path to a pre-trained H-function checkpoint to load instead of training")
    parser.add_argument("--skip_diffusion_training", action="store_true",
                      help="Skip diffusion model training and load from --diffusion_ckpt instead")
    parser.add_argument("--diffusion_ckpt", type=str, default=None,
                      help="Path to a pre-trained diffusion checkpoint to load (required with --skip_diffusion_training)")
    parser.add_argument("--test_days", type=int, default=None,
                      help="Rows to hold out as test set (default: config.TEST_DAYS)")

    args = parser.parse_args()

    train_model(args.data_path, args.seed, args.num_samples, args.gpu, args.epochs, args.save_timesteps, args.skip_hfunction, args.h_ckpt, args.skip_diffusion_training, args.diffusion_ckpt, args.test_days) 
