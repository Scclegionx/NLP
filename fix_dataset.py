import json

# Load dataset
with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("Fixing dataset...")

# Fix train data
for item in data['train']:
    # Remove value field if exists
    if 'value' in item:
        del item['value']
    
    # Fix TIME field for set-alarm
    if item['intent'] == 'set-alarm':
        if 'TIME' in item and item['TIME']:
            item['entities']['TIME'] = item['TIME']
            del item['TIME']
    
    # Ensure all required fields exist in entities
    required_fields = ['RECEIVER', 'MESSAGE', 'PLATFORM', 'DEVICE', 'DATE']
    for field in required_fields:
        if field not in item['entities']:
            item['entities'][field] = ""
    
    # Add specific fields based on intent
    if item['intent'] in ['search-internet', 'search-youtube']:
        if 'QUERY' not in item['entities']:
            item['entities']['QUERY'] = ""
    elif item['intent'] == 'control-device':
        if 'ACTION' not in item['entities']:
            item['entities']['ACTION'] = ""
    elif item['intent'] == 'open-cam':
        if 'MODE' not in item['entities']:
            item['entities']['MODE'] = ""
    elif item['intent'] == 'set-alarm':
        if 'TIME' not in item['entities']:
            item['entities']['TIME'] = ""

# Fix test data
for item in data['test']:
    # Remove value field if exists
    if 'value' in item:
        del item['value']
    
    # Fix TIME field for set-alarm
    if item['intent'] == 'set-alarm':
        if 'TIME' in item and item['TIME']:
            item['entities']['TIME'] = item['TIME']
            del item['TIME']
    
    # Ensure all required fields exist in entities
    required_fields = ['RECEIVER', 'MESSAGE', 'PLATFORM', 'DEVICE', 'DATE']
    for field in required_fields:
        if field not in item['entities']:
            item['entities'][field] = ""
    
    # Add specific fields based on intent
    if item['intent'] in ['search-internet', 'search-youtube']:
        if 'QUERY' not in item['entities']:
            item['entities']['QUERY'] = ""
    elif item['intent'] == 'control-device':
        if 'ACTION' not in item['entities']:
            item['entities']['ACTION'] = ""
    elif item['intent'] == 'open-cam':
        if 'MODE' not in item['entities']:
            item['entities']['MODE'] = ""
    elif item['intent'] == 'set-alarm':
        if 'TIME' not in item['entities']:
            item['entities']['TIME'] = ""

# Save fixed dataset
with open('dataset.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Dataset fixed and saved!")

# Verify
intents = set()
for item in data['train']:
    intents.add(item['intent'])

print(f"Unique intents: {sorted(intents)}")
print(f"Number of intents: {len(intents)}")
print(f"Train samples: {len(data['train'])}")
print(f"Test samples: {len(data['test'])}")
