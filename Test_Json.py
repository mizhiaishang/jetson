import json

with open('caption.json', 'r') as file:
    data = json.load(file)

print(data)
print(data["step1"]["caption"])

for step, content in data.items():
    print(f"Step: {step}")
    print(f"Caption: {content['caption']}")
    print("-----")