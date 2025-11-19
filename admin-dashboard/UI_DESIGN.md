# Violence Detection Dashboard - Ultra Minimalist Design

## Design Philosophy
- **Color Palette**: Monochromatic (Black #000, White #FFF, Gray #F5F5F5 / #333)
- **Typography**: Ultra-thin (100-300 weight), sans-serif, Apple-inspired
- **Spacing**: Generous whitespace, 16px base unit
- **Components**: Outline style, no fill, minimal borders
- **Focus**: Functional clarity over decoration

---

## Color System
```
Primary Background: #FFFFFF
Secondary Background: #F8F8F8
Tertiary Background: #F0F0F0
Text Primary: #000000
Text Secondary: #666666
Border: #E0E0E0
Accent (Violence): #FF4444
Accent (Safe): #00AA00
```

---

## Typography
- **Font Family**: `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif`
- **Weight**: 100 (Thin), 300 (Light), 400 (Regular)
- **Sizes**:
  - H1: 32px, weight 100
  - H2: 24px, weight 100
  - H3: 18px, weight 300
  - Body: 14px, weight 400
  - Caption: 12px, weight 300

---

## Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Dashboard                         [Settings] [Help]       │  ← Header (minimal)
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────────────────────────────────────────────────┐   │
│  │                                                    │   │
│  │            VIDEO STREAM AREA (large)              │   │
│  │                                                    │   │
│  │  Status Badge: Violence / Non-Violence            │   │
│  │  Confidence: 94.2%                                │   │
│  │                                                    │   │
│  └────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Real-time Detection Rate (tiny chart)               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Recent Detections (scrollable list)                │  │
│  │                                                    │  │
│  │ 14:32:45  Violence       Cam 1  94.2%  ●          │  │
│  │ 14:32:10  Non-Violence   Cam 2  45.1%  ○          │  │
│  │ 14:31:55  Violence       Cam 3  87.3%  ●          │  │
│  │                                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Design

### 1. **Header**
```
┌──────────────────────────────────────────────────┐
│ Dashboard              [⚙ Settings] [? Help]    │
└──────────────────────────────────────────────────┘
- Height: 60px
- Padding: 16px horizontal
- Border-bottom: 1px #E0E0E0
- Font: 18px, weight 300
- Icons: outline style, no fill
```

### 2. **Video Container**
```
┌─────────────────────────────────────────────────┐
│                                                 │
│              [VIDEO ELEMENT]                   │
│                                                 │
│  ┌─────────┐  ┌────────────┐  ┌─────────────┐ │
│  │ VIOLENCE│  │ 94.2%      │  │ [DISCONNECT]│ │
│  │ DETECTED│  │ Confidence │  │   OUTLINE  │ │
│  └─────────┘  └────────────┘  └─────────────┘ │
│                                                 │
└─────────────────────────────────────────────────┘
- Aspect ratio: 16:9
- Background: #F8F8F8
- Border: 1px #E0E0E0
- Padding: 24px
- Gap between elements: 16px
```

### 3. **Status Badge**
```
┌──────────────────┐
│ 🔴 VIOLENCE      │  ← When violence detected (red text, no bg)
└──────────────────┘

┌──────────────────┐
│ ✓ NON-VIOLENCE   │  ← When safe (green text, no bg)
└──────────────────┘

- Font: 16px, weight 300
- Padding: 8px 16px
- Border: 1px outline (matches text color)
- Border-radius: 2px (minimal rounding)
```

### 4. **Confidence Meter (Minimal)**
```
Confidence: 94.2%
[━━━━━━━━━━━░░░░░░░]  ← Simple ASCII-like progress bar or just number

- Text only, no fancy progress bar
- OR: Simple outline bar with thin border
```

### 5. **Real-time Chart (Micro)**
```
┌──────────────────────────────────┐
│ Detection Rate (last 5 min)      │
│                                  │
│     ▁▂▃▄▅▆▇█▆▅▄▃▂▁  ← tiny line │
│                                  │
│ Avg: 2.3 det/min                 │
└──────────────────────────────────┘
- Height: 80px (very compact)
- No legend, no grid
- Single thin line
- Minimal axis labels
```

### 6. **Detection Log (Ultra-Clean List)**
```
┌──────────────────────────────────────────────────────┐
│ Recent Detections                                    │
├──────────────────────────────────────────────────────┤
│                                                      │
│ 14:32:45   VIOLENCE      Cam 1   94.2%   ●         │
│ 14:32:10   non-violence   Cam 2   45.1%   ○         │
│ 14:31:55   VIOLENCE      Cam 3   87.3%   ●         │
│ 14:31:20   non-violence   Cam 4   32.0%   ○         │
│ 14:30:45   VIOLENCE      Cam 1   91.5%   ●         │
│                                                      │
└──────────────────────────────────────────────────────┘
- Rows height: 40px
- Columns: Time | Status | Camera | Confidence | Indicator
- Divider: Light gray 1px between rows
- Status: Bold if VIOLENCE, regular if non-violence
- Indicator: Filled circle (●) for violence, empty circle (○) for safe
- Scrollable: max-height 300px
```

### 7. **Buttons (Outline Only)**
```
┌──────────────────┐
│   DISCONNECT     │  ← Outline button
└──────────────────┘

┌──────────────────┐
│   SETTINGS       │  ← Outline button
└──────────────────┘

- Border: 1px #333
- Background: transparent
- Padding: 8px 16px
- Font: 12px, weight 400
- Hover: Background #F8F8F8
- No shadow, no fill
- Border-radius: 2px
```

### 8. **Connection Status (Top Right)**
```
Before: ○ Connecting...
Active: ● Connected
Error:  ● Disconnected

- Dot: 8px diameter
- Text: 12px gray
- No animation (static)
```

---

## Spacing Grid (16px base unit)
```
- Gutters: 24px (1.5 unit)
- Component padding: 16px (1 unit)
- Gap between sections: 32px (2 units)
- Internal element gap: 16px (1 unit)
```

---

## Component Hierarchy

### Page Structure
```
<Dashboard>
  ├── <Header>
  │   ├── Title
  │   ├── Navigation Links (minimal)
  │   └── Icons (settings, help)
  │
  ├── <MainContent>
  │   ├── <VideoSection>
  │   │   ├── Video Element
  │   │   ├── <StatusBadge>
  │   │   ├── <ConfidenceDisplay>
  │   │   └── <DisconnectButton>
  │   │
  │   ├── <ChartSection>
  │   │   └── <DetectionRateChart>
  │   │
  │   └── <LogSection>
  │       └── <DetectionLog>
  │           └── <DetectionRow>[] (scrollable)
```

---

## CSS Architecture

### Reset & Base
```css
* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  font-weight: 400;
  font-size: 14px;
  line-height: 1.6;
  color: #000;
  background: #fff;
}
```

### Semantic Colors
```css
:root {
  --color-bg-primary: #ffffff;
  --color-bg-secondary: #f8f8f8;
  --color-border: #e0e0e0;
  --color-text-primary: #000000;
  --color-text-secondary: #666666;
  --color-danger: #ff4444;
  --color-success: #00aa00;
}
```

---

## Dos & Don'ts

### DO
- ✓ Use generous whitespace
- ✓ Thin font weights (100-300)
- ✓ Outline buttons only
- ✓ One accent color at a time (danger OR success)
- ✓ Minimal borders (1px, light gray)
- ✓ Clear hierarchy through spacing, not colors
- ✓ Monochromatic + single accent color
- ✓ Fast to scan information

### DON'T
- ✗ Gradients
- ✗ Heavy shadows
- ✗ Rounded corners (max 2px)
- ✗ Bright background colors
- ✗ Multiple fonts
- ✗ Animated elements
- ✗ Decorative icons
- ✗ Sidebar navigation

---

## Implementation Priority

1. **Header** - Simple, clean, minimal
2. **Video Container** - Large, breathing room
3. **Status Badge** - Clear, instant understanding
4. **Confidence Display** - Numbers only, no fancy UI
5. **Chart** - Micro, unobtrusive
6. **Detection Log** - Scannable table format
7. **Buttons** - Outline, minimal styling

---

## Responsive Behavior

- **Desktop (>1200px)**: Full layout as designed
- **Tablet (768-1200px)**: Video 100% width, log below
- **Mobile (<768px)**: Stack vertically, hide chart, minimal everything

---

## Font Strategy
- **Headlines**: Weight 100, +4px letter-spacing
- **Body**: Weight 400, normal letter-spacing
- **UI Labels**: Weight 300, 0px letter-spacing

---

## Interaction Design

- **Hover states**: Subtle bg change (#f8f8f8) only
- **Active states**: Border color change, no color fill
- **Transitions**: None (instant feedback)
- **Click feedback**: Border highlight, 100ms

---

## Success Metrics
- User can identify violence status in < 1 second
- No cognitive load from colors or decoration
- Navigation requires 0 thinking
- Information is scannable in 10 seconds
- Vibe: Apple's health app meets minimal dashboard
