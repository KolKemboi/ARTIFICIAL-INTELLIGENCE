class Memory:
    def __init__(self ,memory_limit) -> None:
        self.memory = []
        self.memory_limit = memory_limit

    def add_to_memory(self, usr_input, is_user = True):
        self.memory.append({"message": usr_input, "is_user": is_user})

        if len(self.memory) > self.memory_limit:
            self.memory = self.memory[-self.memory_limit:]

    def get_memory(self):
        return " ".join([f"User: {msg['message']}" if msg['is_user'] else f"Bot: {msg['message']}" for msg in self.memory])