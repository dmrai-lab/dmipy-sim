import json
print(json.dumps({"include": json.load(open("tools/_tier2_timing_plan.json"))}))
