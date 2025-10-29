#!/usr/bin/env python3
"""
CPU Transfer Optimization Demo for Event-Based Depth Inference

This script demonstrates how to use the new CPU transfer optimizations
to reduce the .cpu() bottleneck in the event-based depth inference pipeline.

Usage:
    python cpu_transfer_optimization_demo.py

The optimizations implemented include:
1. Pinned memory buffers for faster GPU-CPU transfers
2. Async transfers using CUDA streams
3. Pre-allocated CPU buffers to avoid memory allocation overhead
4. Direct tensor copy operations when possible
5. Memory pool optimization for faster GPU allocations
6. Detailed profiling of CPU transfer times
"""

import torch
import numpy as np
import time
import sys
import os

# Add the project path to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demo_cpu_transfer_optimization():
    """Demonstrate the CPU transfer optimization improvements."""
    
    print("=== CPU Transfer Optimization Demo ===\n")
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. This demo requires a GPU for meaningful results.")
        return
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name()}\n")
    
    # Test tensor sizes similar to the event camera pipeline
    height, width = 320, 640
    num_iterations = 100
    
    # Create test tensors
    print("Creating test tensors...")
    gpu_tensor = torch.randn(height, width, device=device, dtype=torch.float32)
    
    # Test 1: Standard synchronous CPU transfer
    print(f"\n1. Testing standard synchronous .cpu() transfer ({num_iterations} iterations)...")
    times_sync = []
    
    for i in range(num_iterations):
        start_time = time.time()
        cpu_tensor = gpu_tensor.cpu()
        end_time = time.time()
        times_sync.append(end_time - start_time)
    
    avg_sync = np.mean(times_sync[10:])  # Skip first 10 for warmup
    print(f"   Average sync transfer time: {avg_sync*1000:.2f} ms")
    print(f"   Min: {np.min(times_sync[10:])*1000:.2f} ms, Max: {np.max(times_sync[10:])*1000:.2f} ms")
    
    # Test 2: Async transfer with CUDA streams
    print(f"\n2. Testing async transfer with CUDA streams...")
    transfer_stream = torch.cuda.Stream()
    times_async = []
    
    for i in range(num_iterations):
        with torch.cuda.stream(transfer_stream):
            start_time = time.time()
            cpu_tensor = gpu_tensor.cpu(non_blocking=True)
            transfer_stream.synchronize()
            end_time = time.time()
            times_async.append(end_time - start_time)
    
    avg_async = np.mean(times_async[10:])
    print(f"   Average async transfer time: {avg_async*1000:.2f} ms")
    print(f"   Min: {np.min(times_async[10:])*1000:.2f} ms, Max: {np.max(times_async[10:])*1000:.2f} ms")
    print(f"   Speedup: {avg_sync/avg_async:.2f}x")
    
    # Test 3: Pinned memory transfer
    print(f"\n3. Testing pinned memory buffer transfer...")
    pinned_buffer = torch.empty(height, width, dtype=torch.float32, pin_memory=True)
    times_pinned = []
    
    for i in range(num_iterations):
        with torch.cuda.stream(transfer_stream):
            start_time = time.time()
            pinned_buffer.copy_(gpu_tensor, non_blocking=True)
            transfer_stream.synchronize()
            end_time = time.time()
            times_pinned.append(end_time - start_time)
    
    avg_pinned = np.mean(times_pinned[10:])
    print(f"   Average pinned memory transfer time: {avg_pinned*1000:.2f} ms")
    print(f"   Min: {np.min(times_pinned[10:])*1000:.2f} ms, Max: {np.max(times_pinned[10:])*1000:.2f} ms")
    print(f"   Speedup vs sync: {avg_sync/avg_pinned:.2f}x")
    print(f"   Speedup vs async: {avg_async/avg_pinned:.2f}x")
    
    # Test 4: Complete pipeline simulation (like in dataviewers.py)
    print(f"\n4. Testing complete pipeline simulation...")
    times_pipeline = []
    
    # Simulate the complete processEvents pipeline
    for i in range(num_iterations):
        start_time = time.time()
        
        # Simulate inference output (like seq_events)
        seq_events_sim = torch.randn(1, 1, 10, height, width, device=device)
        
        # Simulate the sum operation and CPU transfer
        img_tensor = torch.sum(seq_events_sim[0][0], dim=0)
        
        # Optimized transfer
        with torch.cuda.stream(transfer_stream):
            pinned_buffer.copy_(img_tensor, non_blocking=True)
            transfer_stream.synchronize()
        
        # Convert to numpy and apply thresholding (like in the real pipeline)
        img_np = pinned_buffer.numpy().astype(np.uint8)
        img_np[img_np != 0] = 255
        
        end_time = time.time()
        times_pipeline.append(end_time - start_time)
    
    avg_pipeline = np.mean(times_pipeline[10:])
    print(f"   Average complete pipeline time: {avg_pipeline*1000:.2f} ms")
    print(f"   Min: {np.min(times_pipeline[10:])*1000:.2f} ms, Max: {np.max(times_pipeline[10:])*1000:.2f} ms")
    
    # Summary
    print(f"\n=== OPTIMIZATION SUMMARY ===")
    print(f"Standard sync transfer:    {avg_sync*1000:.2f} ms")
    print(f"Async transfer:           {avg_async*1000:.2f} ms ({avg_sync/avg_async:.2f}x faster)")
    print(f"Pinned memory transfer:   {avg_pinned*1000:.2f} ms ({avg_sync/avg_pinned:.2f}x faster)")
    print(f"Complete pipeline:        {avg_pipeline*1000:.2f} ms")
    print(f"\nBest optimization achieves {avg_sync/avg_pinned:.2f}x speedup for CPU transfers!")
    
    # Usage instructions
    print(f"\n=== USAGE INSTRUCTIONS ===")
    print("To use these optimizations in your dataviewer:")
    print("1. The optimizations are automatically enabled when setModel() is called")
    print("2. Use dataviewer.enable_cpu_transfer_optimization(True) to explicitly enable")
    print("3. Use dataviewer.get_transfer_performance_stats() to see performance metrics")
    print("4. Use dataviewer.cleanup_resources() when done to free memory")
    
    # Clean up
    transfer_stream.synchronize()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    demo_cpu_transfer_optimization()
