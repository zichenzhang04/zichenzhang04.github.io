"""a Python script that reads a markdown file,identifies the math blocks,
and wraps them with {{< math >}} ... {{< /math >}} brackets.
The script uses regular expressions to find the patterns for inline math ($...$)
and display math ($$...$$)."""

"""Coded by Zichen Zhang, May 23, 2024"""

import re

def wrap_math_blocks(file_path):
    # Read the contents of the markdown file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Define regex patterns for inline and display math
    inline_math_pattern = re.compile(r'\$(.+?)\$')
    display_math_pattern = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)

    # Replace inline math patterns, ensuring non-empty content
    content = inline_math_pattern.sub(lambda m: f"{{< math >}}${m.group(1)}${{< /math >}}" if m.group(1).strip() else m.group(0), content)

    # Replace display math patterns, ensuring non-empty content
    content = display_math_pattern.sub(lambda m: f"{{< math >}}$${m.group(1)}$${{< /math >}}" if m.group(1).strip() else m.group(0), content)

    # Write the modified content back to the markdown file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# File path to the markdown file
file_path = 'index.md'
wrap_math_blocks(file_path)
