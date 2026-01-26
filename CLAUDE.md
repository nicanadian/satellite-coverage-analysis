# CLAUDE.md - Satellite Coverage Analysis

Project-specific instructions for Claude Code when working on this repository.

## Project Overview

Satellite coverage analysis tool that generates PowerPoint presentations with orbital analysis, ground station coverage, and access window visualizations.

## Run Commands

```bash
# Activate virtual environment
source sat-cov-env/bin/activate

# Run analysis with a config (ALWAYS generates full PowerPoint)
python run_analysis.py --config configs/viasat_1sat_30d_apac.yaml

# Skip PowerPoint generation (ONLY use when explicitly requested by user)
python run_analysis.py --config configs/viasat_1sat_30d_apac.yaml --skip-ppt
```

**IMPORTANT**: Always generate the full PowerPoint presentation unless the user explicitly asks to skip it. The `--skip-ppt` flag should only be used for quick validation runs when specifically requested.

## Plot Styling Guidelines

### PowerPoint Slide Dimensions
- Slide size: **13.333 x 7.5 inches** (widescreen 16:9)
- Images placed at: left=0.2", top=0.4", width=12.9"
- Effective image area: ~12.9 x 6.5 inches

### Figure Sizing for Slides
For plots that should fill a slide:
```python
# Full-slide plot (with title space)
fig = plt.figure(figsize=(13, 6))
ax = fig.add_axes([0.05, 0.05, 0.90, 0.82], projection=ccrs.PlateCarree())
ax.set_aspect('auto')  # IMPORTANT: Don't preserve geographic aspect ratio

# Save WITHOUT bbox_inches='tight' to maintain figure dimensions
plt.savefig(output_path, dpi=150, facecolor='white')
```

**CRITICAL**: For map plots, do NOT use `bbox_inches='tight'` - it will crop to content and destroy the aspect ratio. Maps with varying extents will produce tall/narrow images that don't fit slides.

### Multi-Panel Plots
For slides with multiple panels (like regional capacity):
```python
fig = plt.figure(figsize=(16, 10))
ax1 = fig.add_axes([0.03, 0.52, 0.46, 0.43], projection=ccrs.PlateCarree())  # Top left
ax2 = fig.add_axes([0.54, 0.52, 0.43, 0.43])  # Top right
ax3 = fig.add_axes([0.03, 0.05, 0.46, 0.42])  # Bottom left
ax4 = fig.add_axes([0.54, 0.05, 0.43, 0.42])  # Bottom right
```

### Color Conventions
- Satellite track (access window): `#ff7f0e` (orange)
- Track before/after: `gray`, alpha=0.5
- Targets: `red` or `blue` marker='x'
- Ka-band stations: `green`
- TT&C-only stations: `black`
- Passes with access: `green`
- Passes without access: `gray`

### Font Sizes
- Plot title: 11-14pt, bold
- Axis labels: 9-10pt
- Legend: 8-9pt
- Monospace text blocks: 9pt

## Validation & Verification

### After Making Plot Changes
1. Run the analysis
2. Open the generated .pptx file
3. Check each modified slide visually:
   - Does the plot fill the slide appropriately?
   - Is text readable?
   - Are colors correct?
   - Is the legend visible and not overlapping content?

### Screenshot Verification
Screenshots for visual verification should be saved to `/screenshots/` folder.
When asked to evaluate a screenshot:
1. Check if plot fills slide (minimal whitespace)
2. Check aspect ratio matches slide dimensions
3. Check text/legend readability
4. Check that key elements are visible

### Image Dimension Validation
After generating plots, verify dimensions programmatically:
```bash
source sat-cov-env/bin/activate
python -c "from PIL import Image; img = Image.open('results/viasat_1sat_30d_apac/plots/slide_broadside_1.png'); print(f'Size: {img.size}, Aspect ratio: {img.size[0]/img.size[1]:.2f}')"
```
- Aspect ratio should be ~2.0-2.2 for full-slide plots (wider than tall)
- If aspect ratio < 1.5, the image is too tall and won't fit the slide

### Numerical Validation
After analysis runs, verify key metrics in console output:
- Total Accesses count
- Accesses/Day rate
- Valid Contacts count
- TT&C gap statistics

### Config Validation
When modifying configs, ensure:
- `off_nadir_max_deg`: Standard is 60.0
- Ground stations have required fields: lat, lon, min_elevation_deg, ka_band
- Output paths are correctly set

## Key Files

| File | Purpose |
|------|---------|
| `run_analysis.py` | Main entry point, orchestrates analysis |
| `scripts/generate_full_presentation.py` | All plot generation and PowerPoint building |
| `configs/*.yaml` | Mission configuration files |
| `results/*/` | Output directories with .xlsx, .pptx, and plots/ |

## Common Issues

### Plot doesn't fit slide
- Check figure size matches ~13 x 6 inches for full-slide plots
- Use `fig.add_axes([...])` instead of `plt.axes()` for precise positioning
- Use `bbox_inches='tight', pad_inches=0.1` when saving

### Text too small/large
- Adjust fontsize parameter
- Check DPI setting (150 is standard)

### Map projection issues
- Always use `ccrs.PlateCarree()` for consistency
- Set extent after adding all features: `ax.set_extent([lon_min, lon_max, lat_min, lat_max])`
