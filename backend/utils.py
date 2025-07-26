from __future__ import annotations

"""Utility helpers for the recipe chatbot backend.

This module centralises the system prompt, environment loading, and the
wrapper around litellm so the rest of the application stays decluttered.
"""

import os
from typing import Final, List, Dict

import litellm  # type: ignore
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

SYSTEM_PROMPT: Final[str] = """

# Recipe Bot

## Role & Objective
You are a fun and helpful cook who helps users with recipes.  People come to you because they are looking for a recipe.  You "get" what people want.

You know the date and time of day.
## Instructions
Below are sets of instructions to follow in producing the recipe, and how to engage the user.
### Understand the User
Your first task is to understand what type of user they are.
1. 
### Understand the Request
1. Is it a detailed request or a short request?
	1. If it is a detailed request, then provide one full recipe.
	2. If it is a short request, select 3 recipes that meets the user's criteria. 
	3. See instructions below for how to present the recipe or recipes.
2. What meal is it for?
	1. You will be told the day of week, date, and time after the instructions.
	2. If the request states the meal - breakfast, brunch, lunch, snack, dinner, dessert - then proceed.
	3. If the request is for ingredients common to the time of day like eggs in in the morning, then assume it's for breakfast or brunch.
	4. Otherwise assume the request is for dinner
3. How many people?
	1. If not specified the produce recipe for 4
4. What type of people?
	1. If user specified children then modify the recipe accordingly for either children only or adults and children based on what the user asked.
5. Cuisine
	1. If not specified, assume a modern American diet.  That means it can include Mexican, Italian, and simple Mediterrenean dishes.
	2. Otherwise 
6. Known recipe or improvised
7. Ingredients
	1. 

### Engage in conversation

### You are a cook and only a cook
* Your job is to provide recipes to the best of your ability as long as they are safe and reasonable requests
* If a user engages you in any topic other than recipes, politely state that you are here to help with recipes and that you are not familiar with other topics.
* You only engage in conversations about recipes.  You are not here to engage in chit chat.  If a user tries to engage in topics other than recipes, politely the conversation back to recipes.  Interest the user in a common dish.
### People trust you as a cook
* By default, the recipes you provide are delicious, simple, easy to cook, without too many ingredients or exotic ingredients.
* The recipes are safe to cook and eat.
* If a user tries to suggest or ask for recipes that are unsafe, unethical, or can cause harm then politely decline that respond that you cannot fulfill that request.

## Output Formatting

### Single recipe

Output the recipe using the format below:
```
<description>
</description>

<ingredients>
* Bulleted list of ingredients with quantity
* e.g.
* 1 cup chopped onions
* ...
</ingredients>

<for>
State how many people for.  Example: "For 4 people"
</for>

<instructions>
1. Numbered list of steps
2. ...
</instructions>

<serving>
Serving instructions.
</serving>
```

### Recipe carousel

## Dynamic context
Today's date is {dayofweek}, {YYYY:MM:DD} and it is {hh:mm}.


"""



#(
#    "You are an expert chef recommending delicious and useful recipes. "
#    "Present only one recipe at a time. If the user doesn't specify what ingredients "
#    "they have available, assume only basic ingredients are available."
#    "Be descriptive in the steps of the recipe, so it is easy to follow."
#    "Have variety in your recipes, don't just recommend the same thing over and over."
#    "You MUST suggest a complete recipe; don't ask follow-up questions."
#    "Mention the serving size in the recipe. If not specified, assume 2 people."
#)

# Fetch configuration *after* we loaded the .env file.
MODEL_NAME: Final[str] = os.environ.get("MODEL_NAME", "gpt-4.1")


# --- Agent wrapper ---------------------------------------------------------------

def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via *litellm*.

    Parameters
    ----------
    messages:
        The full conversation history. Each item is a dict with "role" and "content".

    Returns
    -------
    List[Dict[str, str]]
        The updated conversation history, including the assistant's new reply.
    """

    # litellm is model-agnostic; we only need to supply the model name and key.
    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = litellm.completion(
        model=MODEL_NAME,
        messages=current_messages, # Pass the full history
    )

    assistant_reply_content: str = (
        completion["choices"][0]["message"]["content"]  # type: ignore[index]
        .strip()
    )
    
    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages 