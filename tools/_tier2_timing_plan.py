import json
plan = json.load(open("tools/_tier2_timing_plan.json"))
print(json.dumps({"include": [{"name": g["name"], "spec": json.dumps(g["spec"])} for g in plan]}))
