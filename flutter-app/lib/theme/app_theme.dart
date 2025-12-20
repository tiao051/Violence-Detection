import 'package:flutter/material.dart';

// =============================================================================
// MODERN MINIMAL DESIGN SYSTEM - Flutter App Theme
// =============================================================================

// -----------------------------------------------------------------------------
// 1. Core Colors (keeping original for compatibility)
// -----------------------------------------------------------------------------
const Color kBackgroundColor = Color(0xFF0F2027); // Dark background
const Color kPrimaryColor = Color(0xFF2B623A); // Green "seed" color

// -----------------------------------------------------------------------------
// 2. Extended Palette - Modern Minimal
// -----------------------------------------------------------------------------
const Color kAccentColor =
    Color(0xFF00D9FF); // Cyan accent - highlights, active states
const Color kAccentSecondary =
    Color(0xFF7B61FF); // Purple accent - secondary highlights
const Color kErrorColor = Color(0xFFFF4757); // Vibrant red - errors, alerts
const Color kSuccessColor = Color(0xFF2ED573); // Success green - confirmations
const Color kWarningColor = Color(0xFFFFB142); // Warning orange

// -----------------------------------------------------------------------------
// 3. Surface Colors - For depth and layering
// -----------------------------------------------------------------------------
const Color kSurfaceColor =
    Color(0xFF1A2634); // Elevated surface (cards, dialogs)
const Color kSurfaceLight =
    Color(0xFF243447); // Lighter surface (hover, selected)
const Color kSurfaceDark = Color(0xFF0D1820); // Darker than background

// -----------------------------------------------------------------------------
// 4. Text Colors
// -----------------------------------------------------------------------------
const Color kTextPrimary = Color(0xFFFFFFFF); // White - primary text
const Color kTextSecondary = Color(0xFFB0BEC5); // Gray - secondary text
const Color kTextMuted = Color(0xFF607D8B); // Muted - disabled, hints

// -----------------------------------------------------------------------------
// 5. Gradients
// -----------------------------------------------------------------------------
const LinearGradient kAppGradient = LinearGradient(
  colors: [kPrimaryColor, kBackgroundColor],
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
);

const LinearGradient kAccentGradient = LinearGradient(
  colors: [kAccentColor, kAccentSecondary],
  begin: Alignment.topLeft,
  end: Alignment.bottomRight,
);

const LinearGradient kSurfaceGradient = LinearGradient(
  colors: [kSurfaceLight, kSurfaceColor],
  begin: Alignment.topCenter,
  end: Alignment.bottomCenter,
);

// -----------------------------------------------------------------------------
// 6. Decoration Helpers - Reusable card/container styles
// -----------------------------------------------------------------------------

/// Standard card decoration with subtle border and shadow
BoxDecoration kCardDecoration({
  Color? color,
  double borderRadius = 16,
  bool hasBorder = true,
  bool hasShadow = true,
}) {
  return BoxDecoration(
    color: color ?? kSurfaceColor,
    borderRadius: BorderRadius.circular(borderRadius),
    border: hasBorder
        ? Border.all(color: Colors.white.withOpacity(0.08), width: 1)
        : null,
    boxShadow: hasShadow
        ? [
            BoxShadow(
              color: Colors.black.withOpacity(0.2),
              blurRadius: 12,
              offset: const Offset(0, 4),
            ),
          ]
        : null,
  );
}

/// Input field decoration with filled style
InputDecoration kInputDecoration({
  required String labelText,
  String? hintText,
  IconData? prefixIcon,
  Widget? suffixIcon,
}) {
  return InputDecoration(
    labelText: labelText,
    hintText: hintText,
    prefixIcon: prefixIcon != null
        ? Icon(prefixIcon, color: kTextSecondary, size: 20)
        : null,
    suffixIcon: suffixIcon,
    filled: true,
    fillColor: kSurfaceColor,
    labelStyle: const TextStyle(color: kTextSecondary),
    hintStyle: TextStyle(color: kTextMuted.withOpacity(0.7)),
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(12),
      borderSide: BorderSide.none,
    ),
    enabledBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(12),
      borderSide: BorderSide(color: Colors.white.withOpacity(0.08)),
    ),
    focusedBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(12),
      borderSide: const BorderSide(color: kAccentColor, width: 1.5),
    ),
    errorBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(12),
      borderSide: const BorderSide(color: kErrorColor),
    ),
    contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
  );
}

/// Standard spacing constants
class AppSpacing {
  static const double xs = 4;
  static const double sm = 8;
  static const double md = 16;
  static const double lg = 24;
  static const double xl = 32;
  static const double xxl = 48;
}

/// Standard border radius constants
class AppRadius {
  static const double sm = 8;
  static const double md = 12;
  static const double lg = 16;
  static const double xl = 24;
  static const double full = 999;
}

// -----------------------------------------------------------------------------
// 7. Main Theme Data
// -----------------------------------------------------------------------------
final ThemeData appThemeData = ThemeData(
  brightness: Brightness.dark,
  useMaterial3: true,

  // Background
  scaffoldBackgroundColor: kBackgroundColor,

  // Color Scheme
  colorScheme: ColorScheme.fromSeed(
    seedColor: kPrimaryColor,
    brightness: Brightness.dark,
    surface: kSurfaceColor,
    error: kErrorColor,
    primary: kPrimaryColor,
    secondary: kAccentColor,
  ),

  // AppBar Theme
  appBarTheme: AppBarTheme(
    backgroundColor: Colors.transparent,
    elevation: 0,
    centerTitle: true,
    titleTextStyle: const TextStyle(
      color: kTextPrimary,
      fontSize: 18,
      fontWeight: FontWeight.w600,
      letterSpacing: 0.5,
    ),
    iconTheme: const IconThemeData(color: kTextPrimary),
  ),

  // Card Theme
  cardTheme: const CardThemeData(
    color: kSurfaceColor,
    elevation: 0,
    margin: EdgeInsets.all(8),
  ),

  // Elevated Button Theme
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      backgroundColor: kPrimaryColor,
      foregroundColor: kTextPrimary,
      elevation: 0,
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(AppRadius.md),
      ),
      textStyle: const TextStyle(
        fontSize: 16,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.5,
      ),
    ),
  ),

  // Text Button Theme
  textButtonTheme: TextButtonThemeData(
    style: TextButton.styleFrom(
      foregroundColor: kAccentColor,
      textStyle: const TextStyle(
        fontSize: 14,
        fontWeight: FontWeight.w500,
      ),
    ),
  ),

  // Input Decoration Theme (default for TextFields)
  inputDecorationTheme: InputDecorationTheme(
    filled: true,
    fillColor: kSurfaceColor,
    labelStyle: const TextStyle(color: kTextSecondary),
    hintStyle: TextStyle(color: kTextMuted.withOpacity(0.7)),
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(AppRadius.md),
      borderSide: BorderSide.none,
    ),
    enabledBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(AppRadius.md),
      borderSide: BorderSide(color: Colors.white.withOpacity(0.08)),
    ),
    focusedBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(AppRadius.md),
      borderSide: const BorderSide(color: kAccentColor, width: 1.5),
    ),
    contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
  ),

  // Chip Theme (for filter chips)
  chipTheme: ChipThemeData(
    backgroundColor: kSurfaceColor,
    selectedColor: kAccentColor.withOpacity(0.2),
    labelStyle: const TextStyle(color: kTextPrimary, fontSize: 13),
    secondaryLabelStyle: const TextStyle(color: kAccentColor, fontSize: 13),
    side: BorderSide(color: Colors.white.withOpacity(0.08)),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(AppRadius.xl),
    ),
    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
  ),

  // ListTile Theme
  listTileTheme: const ListTileThemeData(
    contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
    iconColor: kTextSecondary,
    textColor: kTextPrimary,
  ),

  // Divider Theme
  dividerTheme: DividerThemeData(
    color: Colors.white.withOpacity(0.08),
    thickness: 1,
    space: 1,
  ),

  // Bottom Navigation Bar Theme
  bottomNavigationBarTheme: BottomNavigationBarThemeData(
    backgroundColor: kSurfaceColor,
    selectedItemColor: kAccentColor,
    unselectedItemColor: kTextMuted,
    type: BottomNavigationBarType.fixed,
    elevation: 0,
    selectedLabelStyle:
        const TextStyle(fontSize: 12, fontWeight: FontWeight.w600),
    unselectedLabelStyle: const TextStyle(fontSize: 12),
  ),

  // Snackbar Theme
  snackBarTheme: SnackBarThemeData(
    backgroundColor: kSurfaceLight,
    contentTextStyle: const TextStyle(color: kTextPrimary),
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(AppRadius.md),
    ),
    behavior: SnackBarBehavior.floating,
  ),

  // Dialog Theme
  dialogTheme: DialogThemeData(
    backgroundColor: kSurfaceColor,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(AppRadius.lg),
    ),
    titleTextStyle: const TextStyle(
      color: kTextPrimary,
      fontSize: 20,
      fontWeight: FontWeight.w600,
    ),
  ),
);
