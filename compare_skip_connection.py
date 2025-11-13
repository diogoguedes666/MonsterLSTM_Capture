#!/usr/bin/env python3
"""
Comparison experiment: Skip Connection vs No Skip Connection

This script runs two training experiments:
1. With skip connection (skip_con=1) - baseline
2. Without skip connection (skip_con=0) - test

Both use identical hyperparameters except for skip_con.
Results are saved to separate directories for comparison.
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime
import argparse

def run_training_experiment(name, skip_con, epochs=30, base_args=None):
    """
    Run a training experiment with specified skip connection setting.
    
    Args:
        name: Name identifier for this experiment
        skip_con: Skip connection value (0 or 1)
        epochs: Number of epochs to train
        base_args: Base arguments dictionary (optional)
    
    Returns:
        dict: Experiment results and metadata
    """
    print(f"\n{'='*80}")
    print(f"Starting Experiment: {name}")
    print(f"Skip Connection: {skip_con}")
    print(f"Epochs: {epochs}")
    print(f"{'='*80}\n")
    
    # Build command
    cmd = [
        sys.executable,
        'dist_model_recnet.py',
        '--skip_con', str(skip_con),
        '--epochs', str(epochs),
    ]
    
    # Add base arguments if provided
    if base_args:
        for key, value in base_args.items():
            if key != 'skip_con' and key != 'epochs':  # Don't override our settings
                if isinstance(value, dict):
                    # Handle dictionary arguments (like loss_fcns)
                    # Convert dict to string format that argparse can parse
                    dict_str = '{' + ', '.join([f"'{k}': {v}" for k, v in value.items()]) + '}'
                    cmd.extend([f'--{key}', dict_str])
                elif isinstance(value, bool):
                    if value:
                        cmd.append(f'--{key}')
                else:
                    cmd.extend([f'--{key}', str(value)])
    
    # Set save path to include experiment name
    save_dir = f"Results/compare_skip_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    cmd.extend(['--save_location', save_dir])
    
    print(f"Command: {' '.join(cmd)}\n")
    print(f"Results will be saved to: {save_dir}\n")
    
    # Run training
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Experiment '{name}' completed successfully!")
        print(f"   Time elapsed: {elapsed_time/60:.2f} minutes\n")
        
        # Try to read results
        results = {
            'name': name,
            'skip_con': skip_con,
            'status': 'success',
            'elapsed_time': elapsed_time,
            'save_dir': save_dir
        }
        
        # Try to read training stats
        stats_file = os.path.join(save_dir, 'training_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                stats = json.load(f)
                results['best_val_loss'] = stats.get('best_val_loss', None)
                results['current_epoch'] = stats.get('current_epoch', None)
                results['training_losses'] = stats.get('training_losses', [])
                results['validation_losses'] = stats.get('validation_losses', [])
        
        return results
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Experiment '{name}' failed!")
        print(f"   Time elapsed: {elapsed_time/60:.2f} minutes\n")
        return {
            'name': name,
            'skip_con': skip_con,
            'status': 'failed',
            'elapsed_time': elapsed_time,
            'save_dir': save_dir,
            'error': str(e)
        }


def compare_results(results_with_skip, results_no_skip):
    """
    Compare results from both experiments and print summary.
    """
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    # Status comparison
    print("Experiment Status:")
    print(f"  With Skip (skip_con=1):    {results_with_skip['status']}")
    print(f"  Without Skip (skip_con=0):  {results_no_skip['status']}\n")
    
    if results_with_skip['status'] != 'success' or results_no_skip['status'] != 'success':
        print("⚠️  One or both experiments failed. Check logs above.")
        return
    
    # Training time comparison
    print("Training Time:")
    print(f"  With Skip:    {results_with_skip['elapsed_time']/60:.2f} minutes")
    print(f"  Without Skip: {results_no_skip['elapsed_time']/60:.2f} minutes")
    time_diff = results_no_skip['elapsed_time'] - results_with_skip['elapsed_time']
    if time_diff > 0:
        print(f"  Difference:   +{time_diff/60:.2f} minutes (no skip slower)")
    else:
        print(f"  Difference:   {time_diff/60:.2f} minutes (no skip faster)")
    print()
    
    # Loss comparison
    if 'best_val_loss' in results_with_skip and 'best_val_loss' in results_no_skip:
        print("Best Validation Loss:")
        print(f"  With Skip:    {results_with_skip['best_val_loss']:.6f}")
        print(f"  Without Skip: {results_no_skip['best_val_loss']:.6f}")
        loss_diff = results_no_skip['best_val_loss'] - results_with_skip['best_val_loss']
        if loss_diff > 0:
            print(f"  Difference:   +{loss_diff:.6f} (no skip higher/worse)")
        else:
            print(f"  Difference:   {loss_diff:.6f} (no skip lower/better)")
        print()
    
    # Convergence comparison
    if 'validation_losses' in results_with_skip and 'validation_losses' in results_no_skip:
        val_with = results_with_skip['validation_losses']
        val_without = results_no_skip['validation_losses']
        
        if val_with and val_without:
            print("Convergence Analysis:")
            print(f"  Validation runs (with skip):    {len(val_with)}")
            print(f"  Validation runs (without skip): {len(val_without)}")
            
            if len(val_with) > 0 and len(val_without) > 0:
                final_with = val_with[-1]
                final_without = val_without[-1]
                print(f"  Final validation loss (with):    {final_with:.6f}")
                print(f"  Final validation loss (without): {final_without:.6f}")
            print()
    
    # Save directories
    print("Results Saved To:")
    print(f"  With Skip:    {results_with_skip['save_dir']}")
    print(f"  Without Skip: {results_no_skip['save_dir']}\n")
    
    # Recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80 + "\n")
    
    if 'best_val_loss' in results_with_skip and 'best_val_loss' in results_no_skip:
        if results_no_skip['best_val_loss'] < results_with_skip['best_val_loss']:
            print("✅ No skip connection achieved BETTER validation loss!")
            print("   Consider using skip_con=0 for your final model.\n")
        elif results_with_skip['best_val_loss'] < results_no_skip['best_val_loss']:
            print("✅ Skip connection achieved BETTER validation loss!")
            print("   Consider keeping skip_con=1 for your final model.\n")
        else:
            print("⚠️  Both achieved similar validation loss.")
            print("   Check frequency response plots in dashboard for high-frequency differences.\n")
    
    print("Next Steps:")
    print("1. Check the frequency response plots in both result directories")
    print("2. Listen to best_val_out.wav from both experiments")
    print("3. Compare high-frequency content (15-20kHz) in the dashboard plots")
    print("4. If no skip reduces high-frequency excess, use it for full training\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare skip connection vs no skip connection training'
    )
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs for each experiment (default: 30)')
    parser.add_argument('--data_location', '-dl', default='./Data',
                       help='Location of the Data directory (default: ./Data)')
    parser.add_argument('--file_name', '-fn', default='dls2',
                       help='Filename prefix for input/target files (default: dls2)')
    parser.add_argument('--loss_fcns', default={'ESR': 0.40, 'DC': 0.10, 'HFHinge': 0.30},
                       help='Loss function weights (default: ESR:0.40, DC:0.10, HFHinge:0.30)')
    parser.add_argument('--hf_hinge_strength', type=float, default=1.5,
                       help='HF hinge loss strength (default: 1.5)')
    parser.add_argument('--hf_hinge_fmin', type=float, default=8000,
                       help='HF hinge loss minimum frequency (default: 8000 Hz)')
    parser.add_argument('--hidden_size', '-hs', type=int, default=96,
                       help='Hidden size (default: 96)')
    parser.add_argument('--num_blocks', '-nb', type=int, default=2,
                       help='Number of blocks (default: 2)')
    
    args = parser.parse_args()
    
    # Base arguments for both experiments
    base_args = {
        'data_location': args.data_location,
        'file_name': args.file_name,
        'loss_fcns': args.loss_fcns,
        'hf_hinge_strength': args.hf_hinge_strength,
        'hf_hinge_fmin': args.hf_hinge_fmin,
        'hidden_size': args.hidden_size,
        'num_blocks': args.num_blocks,
        'enable_diagnostics': True,
    }
    
    print("\n" + "="*80)
    print("SKIP CONNECTION COMPARISON EXPERIMENT")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Epochs per experiment: {args.epochs}")
    print(f"  Data location: {args.data_location}")
    print(f"  File name: {args.file_name}")
    print(f"  Loss functions: {args.loss_fcns}")
    print(f"  HF hinge strength: {args.hf_hinge_strength}")
    print(f"  HF hinge fmin: {args.hf_hinge_fmin} Hz")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Number of blocks: {args.num_blocks}")
    print("\nThis will run TWO experiments:")
    print("  1. With skip connection (skip_con=1) - baseline")
    print("  2. Without skip connection (skip_con=0) - test")
    print("\nTotal training time: ~{:.1f} minutes (estimated)".format(args.epochs * 2 * 0.5))
    print("="*80 + "\n")
    
    # Auto-start after 3 seconds, or allow manual start
    import time
    try:
        for i in range(3, 0, -1):
            print(f"Starting in {i} seconds... (Ctrl+C to cancel)", end='\r')
            time.sleep(1)
        print("\n")
    except KeyboardInterrupt:
        print("\n\nCancelled by user.")
        return
    
    # Run experiment 1: With skip connection
    results_with_skip = run_training_experiment(
        name='with_skip',
        skip_con=1,
        epochs=args.epochs,
        base_args=base_args
    )
    
    # Run experiment 2: Without skip connection
    results_no_skip = run_training_experiment(
        name='no_skip',
        skip_con=0,
        epochs=args.epochs,
        base_args=base_args
    )
    
    # Compare results
    compare_results(results_with_skip, results_no_skip)
    
    # Save comparison summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(args),
        'results_with_skip': results_with_skip,
        'results_no_skip': results_no_skip
    }
    
    summary_file = f"Results/compare_skip_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs('Results', exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Full comparison summary saved to: {summary_file}\n")


if __name__ == '__main__':
    main()

