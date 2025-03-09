from src.agentic_patterns.reflection_pattern import ReflectionAgent

agent = ReflectionAgent()

while True:
    try:
        question = input("Prompt: ")
        if question.lower() == "exit":
            print("Exiting...")
            break
        response = agent.run(user_msg=question)
        print("\nResponse:\n", response)
    except Exception as e:
        print(f"Error: {e}")
