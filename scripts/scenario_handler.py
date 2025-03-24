import json

def load_scenario(scenario_id):
    with open('data/scenarios.json') as f:
        scenarios = json.load(f)
        for scenario in scenarios:
            if scenario["scenario_id"] == scenario_id:
                print(f"\nScenario: {scenario['title']}")
                print(f"Description: {scenario['description']}")
                return
        print("Scenario not found.")