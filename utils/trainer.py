import os
import torch
import numpy as np
from tqdm import tqdm
import time

from .utils import get_scores, print_scores, save_scores, save_best_results, timeit, save_model, get_save_model_path


def print_parameter_values(model):
    """Print current values of learnable prompt parameters"""
    print("\n" + "="*60)
    print("PARAMETER VALUES MONITORING")
    print("="*60)
    
    # Check Mode 1 (gamma, beta) parameters
    if hasattr(model.audio_encoder.base, 'gamma') and model.audio_encoder.base.gamma is not None:
        gamma_val = model.audio_encoder.base.gamma.data
        beta_val = model.audio_encoder.base.beta.data
        print(f"\nMode 1 - Simple Parameters:")
        print(f"  gamma shape: {gamma_val.shape}, mean: {gamma_val.mean():.6f}, std: {gamma_val.std():.6f}")
        print(f"  gamma (first 5): {gamma_val[0, 0, :5]}")
        print(f"  beta shape: {beta_val.shape}, mean: {beta_val.mean():.6f}, std: {beta_val.std():.6f}")
        print(f"  beta (first 5): {beta_val[0, 0, :5]}")
    
    # Check Mode 2+ (FiLM) parameters
    if hasattr(model.audio_encoder.base, 'film') and model.audio_encoder.base.film is not None:
        film = model.audio_encoder.base.film
        print(f"\nMode 2+ - FiLM Module:")
        
        # Prompt parameter
        prompt_val = film.prompt.data
        print(f"  Prompt shape: {prompt_val.shape}, mean: {prompt_val.mean():.6f}, std: {prompt_val.std():.6f}")
        print(f"  Prompt (first 5): {prompt_val[0, :5]}")
        
        # MLP weights
        print(f"\n  MLP Architecture:")
        for idx, layer in enumerate(film.mlp):
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.data
                print(f"    Layer {idx} (Linear): in={layer.in_features}, out={layer.out_features}")
                print(f"      Weight - mean: {weight.mean():.6f}, std: {weight.std():.6f}")
                if layer.bias is not None:
                    bias = layer.bias.data
                    print(f"      Bias - mean: {bias.mean():.6f}, std: {bias.std():.6f}")
    
    print("="*60 + "\n")


def run_epoch(model, dataloader, optimizer, criterion, device, args=None, epoch=None, learnable_params=None):
    model.train()

    losses = []
    actual_labels = []
    predicted_labels = []

    for i, (audio, label) in enumerate(dataloader):

        audio = audio.to(device).squeeze(1)
        label = label.to(device)
 

        logits = model(audio)
        loss = criterion(logits, label)
        
        optimizer.zero_grad()
        loss.backward()
        if learnable_params:
            torch.nn.utils.clip_grad_norm_(learnable_params, max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())

        actual_labels.extend(label.cpu().numpy())
        predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())
        
        # Monitor parameter changes during training
        if (i + 1) % 10 == 0:  # Print every 10 batches
            print(f"\n[Epoch {epoch}, Batch {i+1}] Parameter Monitoring:")
            
            # Check Mode 1 (gamma, beta) parameters
            if hasattr(model.audio_encoder, 'gamma') and model.audio_encoder.gamma is not None:
                gamma_val = model.audio_encoder.gamma.data[0, 0, :5]  # First 5 values
                beta_val = model.audio_encoder.beta.data[0, 0, :5]
                print(f"  gamma (first 5): {gamma_val}")
                print(f"  beta (first 5): {beta_val}")
            
            # Check Mode 2+ (FiLM) parameters
            if hasattr(model.audio_encoder, 'film') and model.audio_encoder.film is not None:
                film = model.audio_encoder.film
                # Print prompt
                prompt_val = film.prompt.data[0, :5]  # First 5 values
                print(f"  FiLM prompt (first 5): {prompt_val}")
                # Print first layer of MLP
                first_layer_weight = film.mlp[0].weight.data[0, :5]
                print(f"  FiLM MLP weight (first layer, first row): {first_layer_weight}")

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


@timeit
def run_evaluation(model, dataloader, criterion, device):
    model.eval()

    losses = []
    actual_labels = []
    predicted_labels = []
    
    print("\n\nEvaluating the model ...")
    with torch.no_grad():
        for i, (audio, label) in enumerate(dataloader):
        # for i, (audio, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            print(f"Batch {i+1}/{len(dataloader)}")

            audio = audio.to(device).squeeze(1)
            label = label.to(device)
            
            logits = model(audio)
            loss = criterion(logits, label)

            losses.append(loss.item())

            actual_labels.extend(label.cpu().numpy())
            predicted_labels.extend(logits.argmax(axis=1).cpu().numpy())

    avg_loss = sum(losses) / len(losses)

    return avg_loss, actual_labels, predicted_labels


# @timeit
# def run_evaluation(model, dataloader, criterion, device, num_runs=5):
#     # --- 🚀 GPU 워밍업 (딱 한 번만 강하게) ---
#     if device.type == 'cuda':
#         print("\nWarming up GPU...")
#         dummy_audio, _ = next(iter(dataloader))
#         dummy_audio = dummy_audio.to(device).squeeze(1)
#         with torch.no_grad():
#             for _ in range(5): # 예열을 좀 더 확실하게
#                 _ = model(dummy_audio)
#         torch.cuda.synchronize(device)
#     # ----------------------------------

#     all_runs_latencies = []

#     print(f"\nStarting {num_runs} runs for stable latency measurement...")
    
#     with torch.no_grad():
#         for run in range(num_runs):
#             total_forward_time = 0.0
#             total_samples = 0
            
#             for audio, label in dataloader:
#                 current_batch_size = audio.size(0)
#                 audio = audio.to(device).squeeze(1)
                
#                 if device.type == 'cuda':
#                     torch.cuda.synchronize(device)
#                 start_time = time.perf_counter()
                
#                 logits = model(audio)
                
#                 if device.type == 'cuda':
#                     torch.cuda.synchronize(device)
#                 end_time = time.perf_counter()
                
#                 total_forward_time += (end_time - start_time) * 1000
#                 total_samples += current_batch_size
            
#             # 이번 Run의 1개당 평균 시간 계산
#             run_latency = total_forward_time / total_samples
#             all_runs_latencies.append(run_latency)
#             print(f"Run {run+1}/{num_runs} Latency: {run_latency:.2f} ms")

#     # 💡 평균이 아닌 '중앙값(Median)'을 사용하여 튀는 값 무시!
#     final_stable_latency = np.median(all_runs_latencies)
    
#     print(f"\n✅ Final Stable Per-Sample Latency (Median): {final_stable_latency:.2f} ms")
#     import sys
#     sys.exit()
#     return final_stable_latency


@timeit
def run_training(model, train_dataloader, test_dataloader, optimizer, criterion, device, epochs=50, args=None, learnable_params=None):
    
    # Track best accuracy and corresponding results
    best_accuracy = 0.0
    best_epoch = -1
    best_results = None
    
    for epoch in tqdm(range(epochs), total=epochs):

        train_loss, actual_labels, predicted_labels = run_epoch(model, train_dataloader, optimizer, criterion, device, args=args, epoch=epoch, learnable_params=learnable_params)

        if (epoch+1)%5 == 0:
            train_accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTrain Evaluation (Epoch {epoch + 1}/{epochs})\n-------------------------------\n")
            print_scores(train_accuracy, f1_score, precision, recall, train_loss)
            
            # Print parameter values periodically
            print_parameter_values(model)
            

        if (epoch+1)%args.freq_test_model == 0:
            test_loss, actual_labels, predicted_labels = run_evaluation(model, test_dataloader, criterion, device)
            accuracy, f1_score, precision, recall =  get_scores(actual_labels, predicted_labels, args.classnames)
            print(f"\n\n-------------------------------\nTest Evaluation\n-------------------------------\n")
            print_scores(accuracy, f1_score, precision, recall, test_loss)
            
            # Track best accuracy from test evaluation
            if train_accuracy > best_accuracy:
                best_accuracy = train_accuracy
                best_results = {
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'f1_score': f1_score,
                    'precision': precision,
                    'recall': recall,
                    'test_loss': test_loss,
                    'train accuracy': train_accuracy
                }
                print(f"\n🌟 NEW BEST ACCURACY: {accuracy:.6f} at Epoch {epoch + 1}\n")

            if (epoch == epochs-1) and args.do_logging:
                print("\n\nFinal Evaluation")
                print("Saving Results ...")
                save_scores(args.seed, epoch, accuracy, f1_score, precision, recall, test_loss, args.json_file_path)
                print("Results Saved\n\n")
                
                # Save best results to a separate file
                if best_results is not None:
                    save_best_results(args.seed, best_results, args.json_file_path)
    

    if args.save_model:
        save_model_path = get_save_model_path(args)
        save_model(args, model, save_model_path)
        print(f"Model saved to {save_model_path}")
        