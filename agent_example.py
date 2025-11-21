import os
from groq import Groq
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
GROQ_API_KEY = "API_KEY"
groq_client = Groq(api_key=GROQ_API_KEY)

def get_completion(messages, model="llama-3.1-8b-instant", temperature=0, max_tokens=300, tools=None):
    """Get completion from Groq"""
    response = groq_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        tools=tools
    )
    return response.choices[0].message

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    # Simulate different temperatures for demo purposes
    weather = {
        "location": location,
        "temperature": "22" if unit == "celsius" else "72",
        "unit": unit,
        "condition": "sunny"  # Add weather condition for more realistic response
    }
    return json.dumps(weather)

# Define tools with function schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string", 
                        "enum": ["celsius", "fahrenheit"]
                    },
                },
                "required": ["location"],
            },
        },   
    }
]

def process_conversation(messages):
    """Process a conversation with function calling"""
    # Get the model's response
    response = get_completion(messages, tools=tools)
    
    # Check if the model wants to call a function
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            # Extract function call details
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            # Execute the function
            if func_name == "get_current_weather":
                result = get_current_weather(**func_args)
                
                # Append the function result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": result
                })
                
                # Add a system message to guide the response format
                messages.append({
                    "role": "system",
                    "content": "Please provide a concise response using the weather data. Format: 'The current weather in [location] is [temperature][unit] and [condition].'"
                })
        
        # Get final response from model with function results
        final_response = get_completion(messages)
        return final_response
    
    return response

if __name__ == "__main__":
    print()
    
    while True:
        user_input = input("You: ").strip()
            
        messages = [{"role": "user", "content": user_input}]
        response = process_conversation(messages)
        print("\nAssistant:", response.content)
        print()