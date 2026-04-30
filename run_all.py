import os
import subprocess
import time

# The three loss functions built in Day 4
loss_functions = ['kl', 'jsd', 'custom']

# Ensure a dedicated folder exists for Pavan's logs
log_folder = './logs'
os.makedirs(log_folder, exist_ok=True)

print("🚀 Starting Day 6: Main Training Runs on Local GPU...")
start_time = time.time()

for loss in loss_functions:
    print(f"\n{'='*50}")
    print(f"🔥 INITIATING TRAINING: {loss.upper()} LOSS")
    print(f"{'='*50}")
    
    # Triggering the fine-tuning script Pranav wrote
    command = f"python finetune.py --loss {loss} --log_dir {log_folder}"
    
    try:
        # Run the command and wait for it to finish before starting the next
        subprocess.run(command, shell=True, check=True)
        print(f"✅ Successfully completed {loss.upper()} training.")
    except subprocess.CalledProcessError:
        print(f"❌ ERROR: Training crashed on {loss.upper()} loss. Stopping queue.")
        break

total_time = (time.time() - start_time) / 60
print(f"\n🎉 All Main Training Runs Completed in {total_time:.2f} minutes!")
print(f"📁 Checkpoints saved. Logs are ready in the '{log_folder}' directory for Pavan.")