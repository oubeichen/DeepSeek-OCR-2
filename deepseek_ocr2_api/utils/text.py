"""Text processing utilities."""


def unescape_string(s: str | None) -> str | None:
    """
    Safely unescape common escape sequences in user input.

    Only handles safe, common escape sequences to avoid security risks:
    - \\n -> newline
    - \\t -> tab
    - \\r -> carriage return
    - \\\\ -> backslash

    Args:
        s: Input string that may contain escape sequences

    Returns:
        String with escape sequences converted, or None if input is None
    """
    if s is None:
        return None

    # Process escape sequences manually for safety
    # Only handle common, safe escape sequences
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            next_char = s[i + 1]
            if next_char == 'n':
                result.append('\n')
                i += 2
            elif next_char == 't':
                result.append('\t')
                i += 2
            elif next_char == 'r':
                result.append('\r')
                i += 2
            elif next_char == '\\':
                result.append('\\')
                i += 2
            else:
                # Keep unknown escape sequences as-is
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1

    return ''.join(result)
