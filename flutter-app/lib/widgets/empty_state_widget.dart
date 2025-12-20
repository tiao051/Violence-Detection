import 'package:flutter/material.dart';
import 'package:security_app/theme/app_theme.dart';

/// Reusable widget to display empty state when no data is available
/// Shows icon, title, subtitle, and optional action button
class EmptyStateWidget extends StatelessWidget {
  /// The title to display (e.g., "No Cameras Available")
  final String title;

  /// The subtitle/description message (e.g., "Add cameras in settings...")
  final String subtitle;

  /// Optional icon to display
  final IconData? iconData;

  /// Optional action button text (e.g., "Go to Settings")
  final String? actionButtonText;

  /// Callback when action button is pressed
  final VoidCallback? onActionPressed;

  /// Optional color for the icon (defaults to theme onSurfaceVariant)
  final Color? iconColor;

  /// Show "Pull to refresh" hint text
  final bool showRefreshHint;

  const EmptyStateWidget({
    super.key,
    required this.title,
    required this.subtitle,
    this.iconData,
    this.actionButtonText,
    this.onActionPressed,
    this.iconColor,
    this.showRefreshHint = false,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final effectiveIconColor = iconColor ?? theme.colorScheme.onSurfaceVariant;

    return Center(
      child: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(32.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Large icon
              Icon(
                iconData ?? Icons.inbox_outlined,
                size: 80,
                color: effectiveIconColor,
              ),
              const SizedBox(height: 24),

              // Title
              Text(
                title,
                style: theme.textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                  color: kTextPrimary,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 12),

              // Subtitle
              Text(
                subtitle,
                style: theme.textTheme.bodyMedium?.copyWith(
                  color: kTextSecondary,
                ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 24),

              // Optional action button
              if (actionButtonText != null && onActionPressed != null)
                ElevatedButton(
                  onPressed: onActionPressed,
                  child: Text(actionButtonText!),
                )
              else
                const SizedBox(height: 0),

              // Refresh hint
              if (showRefreshHint) ...[
                const SizedBox(height: 16),
                Text(
                  "Pull down to refresh",
                  style: theme.textTheme.labelSmall?.copyWith(
                    color: kTextMuted,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }
}
