import json
from scenario_handler import load_scenario

def load_menu():
    with open('data/scenario_menu.json') as f:
        menu = json.load(f)
        return menu["scenario_menu"]

def present_menu():
    menu = load_menu()
    print("Please select a scenario to practice:")
    for item in menu:
        print(f"{item['option_id']}. {item['title']}")
    choice = input("Enter the number of your choice: ")
    for item in menu:
        if item["option_id"] == choice:
            load_scenario(item["scenario_id"])
            return
    print("Invalid choice. Try again.")
    present_menu()

present_menu()