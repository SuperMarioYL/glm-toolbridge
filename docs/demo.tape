# VHS tape for glm-toolbridge — renders to assets/demo.gif in CI (demo.yml).
# Shows the before/after: the same OpenAI-format harness against GLM-5.2,
# stalling raw, then completing through one wrap() line.
#
# Render locally (optional):  vhs docs/demo.tape

Output assets/demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 640
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 45ms

Hide
Type "uv sync >/dev/null 2>&1" Enter
Sleep 2s
Type "clear" Enter
Show

Type "# Same OpenAI-format harness, GLM-5.2 backend: raw fails, wrap() fixes it"
Sleep 1500ms
Enter

Type "uv run python examples/openai_harness_demo.py"
Sleep 800ms
Enter
Sleep 5s

Type "# LEFT broke on GLM's native-object arguments; RIGHT parsed after one wrap() line."
Sleep 2s
Enter

Type "# Install:  uv add glm-toolbridge"
Sleep 1500ms
Enter
Sleep 2s
