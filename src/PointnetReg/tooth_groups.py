# tooth_groups.py
# Defines tooth groups and utility functions for tooth ID handling

from typing import List, Optional, Set

# Define tooth groups for orthodontic analysis
TOOTH_GROUPS = {
    "upper": ["t11", "t12", "t13", "t14", "t15", "t16", "t17",
              "t21", "t22", "t23", "t24", "t25", "t26", "t27"],
    "lower": ["t31", "t32", "t33", "t34", "t35", "t36", "t37",
              "t41", "t42", "t43", "t44", "t45", "t46", "t47"],
    "all": ["t11", "t12", "t13", "t14", "t15", "t16", "t17",
            "t21", "t22", "t23", "t24", "t25", "t26", "t27",
            "t31", "t32", "t33", "t34", "t35", "t36", "t37",
            "t41", "t42", "t43", "t44", "t45", "t46", "t47"],
    "upper_right": ["t11", "t12", "t13", "t14", "t15", "t16", "t17"],
    "upper_left": ["t21", "t22", "t23", "t24", "t25", "t26", "t27"],
    "lower_right": ["t41", "t42", "t43", "t44", "t45", "t46", "t47"],
    "lower_left": ["t31", "t32", "t33", "t34", "t35", "t36", "t37"],
}


def get_group_teeth(group_name: str) -> List[str]:
    """
    Get the list of tooth IDs for a given group name.
    
    Args:
        group_name: Name of the tooth group (e.g., "upper", "lower", "all")
        
    Returns:
        List of tooth IDs in the group
        
    Raises:
        ValueError: If group_name is not valid
    """
    if group_name not in TOOTH_GROUPS:
        raise ValueError(f"Unknown tooth group: {group_name}. Available groups: {list(TOOTH_GROUPS.keys())}")
    return TOOTH_GROUPS[group_name]


def get_tooth_group(tooth_id: str) -> Optional[str]:
    """
    Get the group name for a given tooth ID.
    
    Args:
        tooth_id: Tooth ID (e.g., "t11", "t31")
        
    Returns:
        Group name if found, None otherwise
    """
    tooth_id = tooth_id.lower()
    
    # Check which group contains this tooth
    if tooth_id in TOOTH_GROUPS["upper_right"]:
        return "upper_right"
    elif tooth_id in TOOTH_GROUPS["upper_left"]:
        return "upper_left"
    elif tooth_id in TOOTH_GROUPS["lower_right"]:
        return "lower_right"
    elif tooth_id in TOOTH_GROUPS["lower_left"]:
        return "lower_left"
    
    return None


def validate_tooth_id(tooth_id: str) -> bool:
    """
    Validate if a tooth ID is valid.
    
    Args:
        tooth_id: Tooth ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    tooth_id = tooth_id.lower()
    return tooth_id in TOOTH_GROUPS["all"]


def is_valid_tooth_id(tooth_id: str) -> bool:
    """
    Check if a tooth ID is valid (alias for validate_tooth_id).
    
    Args:
        tooth_id: Tooth ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    return validate_tooth_id(tooth_id)


def get_all_tooth_ids() -> List[str]:
    """
    Get all valid tooth IDs.
    
    Returns:
        List of all tooth IDs
    """
    return TOOTH_GROUPS["all"].copy()
