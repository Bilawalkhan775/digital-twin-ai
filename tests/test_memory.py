
from memory import (
    create_conversation,
    save_message,
    get_history,
    delete_conversation
)

# Create two separate conversations
chat1 = create_conversation()
chat2 = create_conversation()

# Visitor 1
save_message(chat1, "user", "My name is Ali")
save_message(chat1, "assistant", "Hello Ali!")

# Visitor 2
save_message(chat2, "user", "My name is Ahmed")
save_message(chat2, "assistant", "Hello Ahmed!")

print("Conversation 1:")
print(get_history(chat1))

print("\nConversation 2:")
print(get_history(chat2))

# Clean up test data
delete_conversation(chat1)
delete_conversation(chat2)
