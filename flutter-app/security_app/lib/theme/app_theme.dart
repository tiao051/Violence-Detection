import 'package:flutter/material.dart';

// 1. Define the 2 "root" colors (constants)
const Color kBackgroundColor = Color(0xFF0F2027); // Dark background color
const Color kPrimaryColor = Color(0xFF2B623A);    // Green "seed" color

// 2. Define the "constant" gradient for consistent use across the app
const LinearGradient kAppGradient = LinearGradient(
  colors: [
    kPrimaryColor,    // Start with green
    kBackgroundColor, // End with dark background
  ],
  begin: Alignment.topLeft,  // Can adjust direction if needed
  end: Alignment.bottomRight,
);

/// Main theme data configuration for the app.
///
/// Uses Material 3 with a dark color scheme based on the primary green color.
/// The scaffold background is set to a custom dark color for consistency.
final ThemeData appThemeData = ThemeData(
  brightness: Brightness.dark, // Enable Dark Mode
  useMaterial3: true,

  // 3. Use the "constant" background color
  scaffoldBackgroundColor: kBackgroundColor,

  colorScheme: ColorScheme.fromSeed(
    seedColor: kPrimaryColor, // Use the "seed" green color
    brightness: Brightness.dark,
    background: kBackgroundColor,
    // Force error color to be deep red instead of pink
    error: Colors.red[700]!,
  ),

  // 4. Tweak theme for AppBar (to enable "auto" gradient if needed later)
  appBarTheme: const AppBarTheme(
    backgroundColor: Colors.transparent, // Important: Make AppBar transparent to show gradient
    elevation: 0,
    // If you want gradient "auto" for ALL AppBars, you can use flexibleSpace:
    // flexibleSpace: Container(decoration: const BoxDecoration(gradient: kAppGradient)),
  ),

  // 5. Tweak theme for Buttons (to enable "auto" gradient if needed later)
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      // Can set "default" (standard) style here
      // For example: padding, shape, etc.
    ),
  ),
);
