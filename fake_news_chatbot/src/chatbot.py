import sys
import os

# Add src directory to path to import predict module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict import analyze_news

print("🔥 Fake News Chatbot Ready! Type 'exit' to quit.\n")

while True:
    try:
        user_input = input("Enter a news sentence: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input.strip():
            print("Please enter a non-empty sentence.\n")
            continue

        output = analyze_news(user_input)
        print(f"\nAnalysis: {output['result']}")
        print(f"Confidence: {output['confidence']}%\n")
    
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}\n")
