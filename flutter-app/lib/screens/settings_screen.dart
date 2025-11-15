import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/auth_provider.dart';
import '../providers/camera_provider.dart';
import '../providers/event_provider.dart';
import '../providers/settings_provider.dart';
import '../theme/app_theme.dart';

/// Settings screen for app configuration and preferences.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  @override
  void initState() {
    super.initState();
    // Load settings when screen opens
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<SettingsProvider>().loadSettings();
    });
  }

  /// Show confirm dialog for destructive actions
  void _showConfirmDialog(
    String title,
    String message,
    VoidCallback onConfirm,
  ) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              onConfirm();
            },
            child: Text('Confirm', style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ),
        ],
      ),
    );
  }

  /// Handle clear cache action
  void _handleClearCache() {
    _showConfirmDialog(
      'Clear Cache',
      'This will clear all cached data. Continue?',
      () {
        context.read<CameraProvider>().clearCache();
        context.read<EventProvider>().clearCache();
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Cache cleared successfully'),
            backgroundColor: Colors.green,
          ),
        );
      },
    );
  }

  /// Handle logout action
  void _handleLogout() {
    _showConfirmDialog(
      'Logout',
      'Are you sure you want to logout?',
      () {
        context.read<SettingsProvider>().clearCache();
        context.read<CameraProvider>().clearCache();
        context.read<EventProvider>().clearCache();
        context.read<AuthProvider>().logout();
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        flexibleSpace: Container(
          decoration: const BoxDecoration(gradient: kAppGradient),
        ),
      ),
      body: Consumer<SettingsProvider>(
        builder: (context, settingsProvider, child) {
          return ListView(
            children: [
              // SECURITY SECTION
              _buildSectionHeader('Security'),
              _buildSwitchTile(
                title: 'Enable Motion Alerts',
                subtitle: 'Receive alerts when motion is detected',
                value: settingsProvider.enableMotionAlerts,
                onChanged: (value) {
                  settingsProvider.updateSetting('enableMotionAlerts', value);
                },
              ),
              _buildSliderTile(
                title: 'Alert Sensitivity',
                subtitle: '${settingsProvider.alertSensitivity}%',
                value: settingsProvider.alertSensitivity.toDouble(),
                onChanged: (value) {
                  settingsProvider.updateSetting('alertSensitivity', value.toInt());
                },
                min: 0,
                max: 100,
                divisions: 10,
              ),
              _buildDropdownTile(
                title: 'Alert Sound',
                subtitle: _getAlertSoundLabel(settingsProvider.alertSound),
                value: settingsProvider.alertSound,
                items: ['silent', 'vibrate', 'sound'],
                itemLabels: ['Silent', 'Vibrate', 'Sound'],
                onChanged: (value) {
                  if (value != null) {
                    settingsProvider.updateSetting('alertSound', value);
                  }
                },
              ),
              const Divider(),

              // DISPLAY SECTION
              _buildSectionHeader('Display'),
              _buildSwitchTile(
                title: 'Dark Mode',
                subtitle: 'Use dark theme (currently always enabled)',
                value: settingsProvider.darkMode,
                onChanged: (value) {
                  settingsProvider.updateSetting('darkMode', value);
                },
                enabled: false, // Dark mode always on for this app
              ),
              _buildDropdownTile(
                title: 'Refresh Rate',
                subtitle: '${settingsProvider.refreshRateSeconds}s',
                value: settingsProvider.refreshRateSeconds.toString(),
                items: ['1', '5', '10', '30'],
                itemLabels: ['1 second', '5 seconds', '10 seconds', '30 seconds'],
                onChanged: (value) {
                  if (value != null) {
                    settingsProvider.updateSetting('refreshRateSeconds', int.parse(value));
                  }
                },
              ),
              const Divider(),

              // NOTIFICATIONS SECTION
              _buildSectionHeader('Notifications'),
              _buildSwitchTile(
                title: 'Enable Push Notifications',
                subtitle: 'Receive FCM push notifications',
                value: settingsProvider.enableNotifications,
                onChanged: (value) {
                  settingsProvider.updateSetting('enableNotifications', value);
                },
              ),
              _buildSwitchTile(
                title: 'Show Badge',
                subtitle: 'Display unread event badge',
                value: settingsProvider.showBadge,
                onChanged: (value) {
                  settingsProvider.updateSetting('showBadge', value);
                },
              ),
              const Divider(),

              // ABOUT SECTION
              _buildSectionHeader('About'),
              _buildSimpleTile(
                title: 'App Version',
                subtitle: '1.0.0',
              ),
              _buildActionTile(
                title: 'Terms & Conditions',
                onTap: () {
                  _showInfoDialog('Terms & Conditions', 'App terms and conditions go here.');
                },
              ),
              _buildActionTile(
                title: 'Privacy Policy',
                onTap: () {
                  _showInfoDialog('Privacy Policy', 'Our privacy policy goes here.');
                },
              ),
              _buildActionTile(
                title: 'About Us',
                onTap: () {
                  _showInfoDialog('About Us', 'Security app for violence detection v1.0.0');
                },
              ),
              const Divider(),

              // ACTION BUTTONS
              _buildSectionHeader('Actions'),
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    ElevatedButton.icon(
                      onPressed: _handleClearCache,
                      icon: const Icon(Icons.delete_outline),
                      label: const Text('Clear Cache'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.orange,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                    const SizedBox(height: 12),
                    ElevatedButton.icon(
                      onPressed: _handleLogout,
                      icon: const Icon(Icons.logout),
                      label: const Text('Logout'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Theme.of(context).colorScheme.error,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
            ],
          );
        },
      ),
    );
  }

  /// Build section header widget
  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 8),
      child: Text(
        title.toUpperCase(),
        style: Theme.of(context).textTheme.labelLarge?.copyWith(
          color: Theme.of(context).colorScheme.primary,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  /// Build switch tile
  Widget _buildSwitchTile({
    required String title,
    required String subtitle,
    required bool value,
    required Function(bool) onChanged,
    bool enabled = true,
  }) {
    return SwitchListTile(
      title: Text(title),
      subtitle: Text(subtitle, maxLines: 1, overflow: TextOverflow.ellipsis),
      value: value,
      onChanged: enabled ? onChanged : null,
    );
  }

  /// Build slider tile
  Widget _buildSliderTile({
    required String title,
    required String subtitle,
    required double value,
    required Function(double) onChanged,
    required double min,
    required double max,
    int? divisions,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.w500)),
          Text(subtitle, style: Theme.of(context).textTheme.bodySmall),
          Slider(
            value: value,
            min: min,
            max: max,
            divisions: divisions,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }

  /// Build dropdown tile
  Widget _buildDropdownTile({
    required String title,
    required String subtitle,
    required String value,
    required List<String> items,
    required List<String> itemLabels,
    required Function(String?) onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.w500)),
          DropdownButton<String>(
            value: value,
            isExpanded: true,
            items: List.generate(
              items.length,
              (index) => DropdownMenuItem(
                value: items[index],
                child: Text(itemLabels[index]),
              ),
            ),
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }

  /// Build simple text tile
  Widget _buildSimpleTile({
    required String title,
    required String subtitle,
  }) {
    return ListTile(
      title: Text(title),
      subtitle: Text(subtitle),
      enabled: false,
    );
  }

  /// Build action tile
  Widget _buildActionTile({
    required String title,
    required VoidCallback onTap,
  }) {
    return ListTile(
      title: Text(title),
      trailing: const Icon(Icons.chevron_right),
      onTap: onTap,
    );
  }

  /// Show info dialog
  void _showInfoDialog(String title, String content) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: SingleChildScrollView(
          child: Text(content),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  /// Get readable label for alert sound
  String _getAlertSoundLabel(String sound) {
    switch (sound) {
      case 'silent':
        return 'Silent';
      case 'vibrate':
        return 'Vibrate';
      case 'sound':
        return 'Sound';
      default:
        return 'Sound';
    }
  }
}
