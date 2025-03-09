import json
import requests
from src.agentic_patterns.reflection_pattern import ReflectionAgent
from src.agentic_patterns.tool_pattern import ToolAgent
from src.agentic_patterns.tool_pattern import tool

choice = int(input("Enter your choice: "))
if choice==1:
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
elif choice==2:
    def fetch_top_hacker_news_stories(top_n: int):
        top_stories_url='https://hacker-news.firebaseio.com/v0/topstories.json'

        try:
            response = requests.get(top_stories_url)
            response.raise_for_status()
            top_story_ids = response.json()[:top_n]
            top_stories=[]

            for story_id in top_story_ids:
                story_url = f'https://hacker-news.firebaseio.com/v0/item/{story_id}.json'
                story_response = requests.get(story_url)
                story_response.raise_for_status()
                story_data = story_response.json()
                top_stories.append({
                    'title': story_data.get('title','No title'),
                    'url': story_data.get('url','No URL available')
                })
            return json.dumps(top_stories)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return []
        
    hn_tool = tool(fetch_top_hacker_news_stories)
    agent = ToolAgent([hn_tool],"llama3-70b-8192")
    user_msg = input("What would you like to do?")
    response = agent.run(user_msg)
    print(f"\n{response}")
