import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

/// Reusable error widget with icon, message, and retry button
/// Used when data fetch fails or video loading fails
class ErrorWidget extends StatelessWidget {
  /// The error message to display
  final String errorMessage;

  /// Callback when retry button is pressed
  /// Should call fetchCameras(), fetchEvents(), or _initializePlayer()
  final VoidCallback onRetry;

  /// Optional custom icon, defaults to warning icon
  final IconData? iconData;

  /// Optional custom title, defaults to "Oops!"
  final String? title;

  const ErrorWidget({
    super.key,
    required this.errorMessage,
    required this.onRetry,
    this.iconData,
    this.title,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final errorColor = theme.colorScheme.error;

    return Center(
      child: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Icon with gradient background
              Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: LinearGradient(
                    colors: [
                      errorColor.withOpacity(0.3),
                      errorColor.withOpacity(0.1),
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
                child: Icon(
                  iconData ?? FontAwesomeIcons.exclamation,
                  color: errorColor,
                  size: 40,
                ),
              ),
              const SizedBox(height: 20),

              // Title
              Text(
                title ?? "Oops!",
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: theme.colorScheme.onSurface,
                ),
              ),
              const SizedBox(height: 12),

              // Error message
              Text(
                errorMessage,
                textAlign: TextAlign.center,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
              const SizedBox(height: 32),

              // Retry button with gradient
              Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      errorColor.withOpacity(0.8),
                      errorColor,
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: ElevatedButton.icon(
                  style: ElevatedButton.styleFrom(
                    minimumSize: const Size(200, 50),
                    backgroundColor: Colors.transparent,
                    foregroundColor: Colors.white,
                    shadowColor: Colors.transparent,
                    elevation: 0,
                  ),
                  onPressed: onRetry,
                  icon: const Icon(Icons.refresh),
                  label: const Text("Retry"),
                ),
              ),
              const SizedBox(height: 16),

              // Helpful text
              Text(
                "Please check your internet connection\nand try again.",
                textAlign: TextAlign.center,
                style: theme.textTheme.labelSmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
