import json
import os
from glob import glob

data_dir = 'preprocess/femnist/data_niid'

print("Checking what folders exist...")
print(os.listdir(data_dir))
print()

train_dir = os.path.join(data_dir, 'train')
if os.path.exists(train_dir):
    print(f"Train folder exists!")
    train_files = glob(os.path.join(train_dir, '*.json'))
    print(f"Found {len(train_files)} JSON files in train/")
    
    if train_files:
        print(f"\nLoading: {train_files[0]}")
        with open(train_files[0], 'r') as f:
            data = json.load(f)
        
        print(f"\nKeys in JSON: {data.keys()}")
        print(f"Number of users: {len(data['users'])}")
        print(f"First few users: {data['users'][:3]}")
        
        first_user = data['users'][0]
        user_data = data['user_data'][first_user]
        
        print(f"\nFirst user: {first_user}")
        print(f"Number of samples: {len(user_data['x'])}")
        print(f"Image shape (should be 784): {len(user_data['x'][0])}")
        print(f"First few labels: {user_data['y'][:10]}")
        print(f"Label range: {min(user_data['y'])} to {max(user_data['y'])}")
else:
    print(f"Train folder doesn't exist!")
    print("Available folders:")
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            print(f"  - {item}/")
            json_files = glob(os.path.join(item_path, '*.json'))
            if json_files:
                print(f"    ({len(json_files)} JSON files)")