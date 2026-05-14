---
type: snippet
title: "Rich Console Output"
libs:
  - "other"
tags:
  - "task/visualization"
related:
  - "[[Pandas Plotting]]"
created: 2026-05-13
updated: 2026-05-13
---

# Rich Console Output

> The [Rich](https://github.com/Textualize/rich) library provides beautiful terminal formatting — colors, tables, progress bars, syntax highlighting, and more. This note covers color discovery and common output patterns.

---

## List All Named Colors

> Print every built-in ANSI color name, rendered in its own color. `ANSI_COLOR_NAMES` is a dict mapping color name → ANSI code.

```python
from rich.color import ANSI_COLOR_NAMES
from rich.console import Console

console = Console()
text = "  ".join(f"[{name}]{name}[/{name}]" for name in sorted(ANSI_COLOR_NAMES))
console.print(text)
```

**Gotchas:** These are the 256 standard ANSI color names. Actual rendering depends on your terminal's color scheme — the same name can look different in different terminals.

---

## Inspect the Color Palette Programmatically

> Get color names as a list for filtering, counting, or building dynamic styles.

```python
from rich.color import ANSI_COLOR_NAMES

# Total number of named colors
print(f"Total named colors: {len(ANSI_COLOR_NAMES)}")

# Filter colors by substring
blues = [name for name in sorted(ANSI_COLOR_NAMES) if "blue" in name]
reds = [name for name in sorted(ANSI_COLOR_NAMES) if "red" in name]
greens = [name for name in sorted(ANSI_COLOR_NAMES) if "green" in name]

print(f"Blues:  {blues}")
print(f"Reds:   {reds}")
print(f"Greens: {greens}")
```

**Gotchas:** `ANSI_COLOR_NAMES` is a dict of `{name: number}`. The number is the ANSI 256-color index, not an RGB value.

---

## Rich Color Systems

> Rich supports multiple color systems beyond named ANSI colors. You can use hex codes, RGB tuples, and the full 256-color palette.

```python
from rich.console import Console
from rich.text import Text

console = Console()

# Named color
console.print("[magenta]Named color: magenta[/magenta]")

# Hex color
console.print("[#ff6347]Hex color: #ff6347 (tomato)[/#ff6347]")

# RGB color
console.print("[rgb(100,200,50)]RGB color: rgb(100,200,50)[/rgb(100,200,50)]")

# Color on background
console.print("[bold white on dark_red] White text on dark red background [/bold white on dark_red]")

# Combine styles
console.print("[bold italic underline bright_cyan]All the styles at once![/bold italic underline bright_cyan]")
```

**Gotchas:** Not all terminals support true-color (24-bit). Rich auto-detects and downgrades gracefully, but colors may appear different on legacy terminals. Use `Console(force_terminal=True)` to override detection.

---

## Color Swatches Table

> Render a visual swatch table to compare colors side by side — useful for picking a palette.

```python
from rich.color import ANSI_COLOR_NAMES
from rich.console import Console
from rich.table import Table

console = Console()
table = Table(title="Rich Color Swatches", show_lines=False)
table.add_column("Color Name", style="bold")
table.add_column("Sample", width=20)
table.add_column("ANSI Code", justify="right")

for name in sorted(ANSI_COLOR_NAMES):
    code = ANSI_COLOR_NAMES[name]
    table.add_row(name, f"[{name}]{'█' * 16}[/{name}]", str(code))

console.print(table)
```

**Gotchas:** Printing all ~256 colors creates a long table. Pipe to a pager or filter by substring (see pattern above) for a shorter view.

---

## Using Rich Themes for Consistent Styling

> Define a reusable theme so your ML scripts have consistent color coding for info, warnings, results, etc.

```python
from rich.console import Console
from rich.theme import Theme

ml_theme = Theme({
    "info": "cyan",
    "warning": "yellow bold",
    "error": "red bold",
    "metric": "green",
    "epoch": "magenta bold",
    "param": "bright_blue",
})

console = Console(theme=ml_theme)

console.print("[info]Loading dataset...[/info]")
console.print("[warning]Missing values detected in 3 columns[/warning]")
console.print("[epoch]Epoch 10/100[/epoch]  [metric]accuracy=0.943[/metric]  [metric]loss=0.217[/metric]")
console.print("[param]lr=0.001[/param]  [param]batch_size=64[/param]")
console.print("[error]CUDA out of memory![/error]")
```

**Gotchas:** Theme names are arbitrary strings but cannot conflict with Rich's built-in style names like `bold`, `italic`, etc.

---
