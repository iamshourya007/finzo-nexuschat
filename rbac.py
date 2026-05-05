"""Role-Based Access Control configuration."""

def get_role_filter(role: str) -> dict:
    """
    Return ChromaDB metadata filter for a given role.
    """
    if role == "cxo_level":
        return {} # No filter, full access
    
    if role == "employee":
        return {"department": {"$in": ["general"]}}
    
    return {"department": {"$in": [role, "general"]}}
