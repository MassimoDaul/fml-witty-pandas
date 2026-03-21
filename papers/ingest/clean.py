import re

# Strip display math: $$...$$ and \[...\]
_DISPLAY_MATH = re.compile(r'\$\$.*?\$\$|\\\[.*?\\\]', re.DOTALL)

# Strip inline math: $...$
_INLINE_MATH = re.compile(r'\$[^$\n]*?\$')

# \textbf{X} -> X, \emph{X} -> X, etc. (keep inner content)
_LATEX_STYLED = re.compile(
    r'\\(?:textbf|emph|textit|text|mathrm|mathbf|mathit|boldsymbol|'
    r'hat|bar|tilde|vec|dot|ddot|underline|overline)\{([^}]*)\}'
)

# \cite{...}, \ref{...}, \label{...}, \begin{...}, \end{...}, etc. -> remove entirely
_LATEX_COMMAND = re.compile(r'\\[a-zA-Z]+\{[^}]*\}')

# Bare backslash commands: \alpha, \to, etc.
_LATEX_BARE = re.compile(r'\\[a-zA-Z]+')

# Leftover braces
_BRACES = re.compile(r'[{}]')

_WHITESPACE = re.compile(r'\s+')


def clean_text(text: str) -> str:
    text = text.replace('\n', ' ')
    text = _DISPLAY_MATH.sub(' ', text)
    text = _INLINE_MATH.sub(' ', text)
    text = _LATEX_STYLED.sub(r'\1', text)
    text = _LATEX_COMMAND.sub(' ', text)
    text = _LATEX_BARE.sub(' ', text)
    text = _BRACES.sub('', text)
    text = _WHITESPACE.sub(' ', text).strip()
    return text


def build_embed_input(title: str, abstract: str) -> str:
    """Prepend the task prefix required by nomic-embed-text-v1.5."""
    return f"search_document: {clean_text(title)}. {clean_text(abstract)}"
