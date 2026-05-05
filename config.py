"""Configuration settings and user definitions."""

# Users definition (Username -> Details)
USERS = {
    "Lewis": {
        "password": "password123",
        "email": "Lewis@finzo",
        "role": "cxo_level",
        "name": "Lewis Hamilton",
        "title": "CIO"
    },
    "Max": {
        "password": "password123",
        "email": "max@hr",
        "role": "hr",
        "name": "Max Verstappen",
        "title": "HR Manager"
    },
    "Kimi": {
        "password": "password123",
        "email": "kimi@finance",
        "role": "finance",
        "name": "Kimi Antonelli",
        "title": "Finance Analyst"
    },
    "George": {
        "password": "password123",
        "email": "george@mkt",
        "role": "marketing",
        "name": "George Russel",
        "title": "Marketing Manager"
    },
    "Charles": {
        "password": "password123",
        "email": "charles@eng",
        "role": "engineering",
        "name": "Charles Leclerc",
        "title": "Senior Engineer"
    },
    "Lando": {
        "password": "password123",
        "email": "Lando@finzo",
        "role": "employee",
        "name": "Lando Norris",
        "title": "Associate"
    }
}

# Role display mapping (Role -> (Emoji, Display Name))
ROLE_DISPLAY = {
    "cxo_level": ("👑", "CXO Level"),
    "hr": ("👥", "HR"),
    "finance": ("💰", "Finance"),
    "marketing": ("📈", "Marketing"),
    "engineering": ("💻", "Engineering"),
    "employee": ("👤", "Employee")
}

CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
