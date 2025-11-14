# HealthForecast AI Brand Assets

This directory contains brand assets for the HealthForecast AI application.

## Files

### `brand_logo.svg` ✅
Primary logo file featuring:
- HF monogram in a rounded hexagon shape
- Electric blue to aqua gradient (#60A5FA → #22D3EE)
- Magenta accent stroke (#F472B6)
- Motion lines suggesting forecasting
- Outer glow effect
- 256x256 artboard

**Status**: Created and ready to use

### `brand_logo_128.png` (Optional)
PNG fallback for environments that don't support SVG.

**To create**:
1. Open `brand_logo.svg` in a vector editor (Inkscape, Illustrator, Figma, etc.)
2. Export as PNG at 128x128px
3. Save as `brand_logo_128.png` in this directory

**Note**: The app will work without this file. The SVG will be used by default, and if the SVG is unavailable, a programmatic fallback is generated.

### `favicon.png` (Optional)
64x64 favicon for browser tabs.

**To create**:
1. Open `brand_logo.svg` in a vector editor
2. Export as PNG at 64x64px
3. Save as `favicon.png` in this directory
4. Optionally convert to `.ico` format for broader browser support

**Note**: The favicon is optional and doesn't affect the sidebar branding.

## Usage

The brand is automatically rendered at the top of the sidebar via:
- `render_sidebar_brand()` - Renders the logo and text
- `inject_brand_css()` - Applies Bolt/Raycast-inspired styling

Both functions are called in `app.py` and will gracefully fall back if assets are missing.

## Styling

The brand uses:
- **Glassmorphism**: Frosted backdrop with blur and saturation
- **Gradient text**: Electric blue → aqua → magenta for the title
- **Sticky positioning**: Logo stays visible when scrolling
- **Responsive**: Text hides on narrow widths (<380px)
- **Neon glow**: Soft cyan shadow on the logo

## Color Palette

| Color | Hex | Usage |
|-------|-----|-------|
| Electric Blue | #60A5FA | Primary gradient start |
| Aqua | #22D3EE | Primary gradient middle |
| Magenta | #F472B6 | Accent stroke, gradient end |
| Text Strong | #E6ECFF | Primary text |
| Text Subtle | #9FB3D1 | Secondary text/tagline |
| Card Background | rgba(16,23,42,0.85) | Glassmorphic surfaces |

## Customization

To customize the brand:

1. **Change logo**: Replace `brand_logo.svg` with your own design (keep 256x256 artboard)
2. **Modify text**: Edit the HTML in `render_sidebar_brand()` in `app_core/ui/components.py`
3. **Adjust styling**: Update CSS in `inject_brand_css()` in the same file
4. **Change colors**: Update the gradient values in both functions

## Technical Details

- SVG is embedded as a data URI to avoid file serving issues
- URL encoding is applied for proper browser rendering
- Fallback chain: SVG → PNG → Programmatic SVG → No-op
- Compatible with Streamlit's unsafe_allow_html disabled (graceful degradation)
