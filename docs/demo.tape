# PaperMind demo recording — deterministic, no API key needed.
#
# 1. Install VHS:  https://github.com/charmbracelet/vhs
#       brew install vhs      (macOS)
#       scoop install vhs     (Windows)  /  see repo for Linux
# 2. Install PaperMind:  pip install -e .
# 3. Render:  vhs docs/demo.tape   ->  produces docs/demo.gif
#
# `papermind demo` is self-paced (~12s) and clears the screen between the two
# beats (analyze -> ask), so each beat gets a clean, readable frame.

Output docs/demo.gif
Require papermind

Set Shell bash
Set FontSize 16
Set Width 1200
Set Height 1000
Set Padding 20
Set Theme "Dracula"

Hide
Type "clear"
Enter
Show

Type "papermind demo"
Sleep 600ms
Enter
Sleep 13s
