"""Memory module — conversation history + follow-up generation."""
from app.memory.conversation_memory import (
    ConversationMemory,
    get_conversation_memory,
    add_user_message,
    add_assistant_message,
    get_history,
    get_formatted_history,
)
from app.memory.followup_generator import (
    FollowUpGenerator,
    generate_followup_questions,
)
