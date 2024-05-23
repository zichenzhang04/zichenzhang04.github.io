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
    inline_math_pattern = re.compile(r'\$(.*?)\$')
    display_math_pattern = re.compile(r'\$\$(.*?)\$\$')

    # Replace inline math patterns
    content = inline_math_pattern.sub(r'{{< math >}}$\1${{< /math >}}', content)

    # Replace display math patterns
    content = display_math_pattern.sub(r'{{< math >}}$$\1$${{< /math >}}', content)

    # Write the modified content back to the markdown file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

# File path to the markdown file
file_path = 'index.md'
wrap_math_blocks(file_path)
