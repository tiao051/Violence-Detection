/// Model representing app settings and user preferences.
class SettingsModel {
  /// Security: Enable motion detection alerts
  final bool enableMotionAlerts;

  /// Security: Alert sensitivity level (0-100%)
  final int alertSensitivity;

  /// Security: Alert notification sound type
  final String alertSound; // "silent", "vibrate", "sound"

  /// Display: Dark mode toggle
  final bool darkMode;

  /// Display: Event refresh rate in seconds
  final int refreshRateSeconds;

  /// Notifications: Enable push notifications
  final bool enableNotifications;

  /// Notifications: Show unread badge
  final bool showBadge;

  SettingsModel({
    this.enableMotionAlerts = true,
    this.alertSensitivity = 70,
    this.alertSound = "sound",
    this.darkMode = true,
    this.refreshRateSeconds = 5,
    this.enableNotifications = true,
    this.showBadge = true,
  });

  /// Convert SettingsModel to JSON for persistence
  Map<String, dynamic> toJson() {
    return {
      'enableMotionAlerts': enableMotionAlerts,
      'alertSensitivity': alertSensitivity,
      'alertSound': alertSound,
      'darkMode': darkMode,
      'refreshRateSeconds': refreshRateSeconds,
      'enableNotifications': enableNotifications,
      'showBadge': showBadge,
    };
  }

  /// Convert JSON to SettingsModel
  factory SettingsModel.fromJson(Map<String, dynamic> json) {
    return SettingsModel(
      enableMotionAlerts: json['enableMotionAlerts'] as bool? ?? true,
      alertSensitivity: json['alertSensitivity'] as int? ?? 70,
      alertSound: json['alertSound'] as String? ?? "sound",
      darkMode: json['darkMode'] as bool? ?? true,
      refreshRateSeconds: json['refreshRateSeconds'] as int? ?? 5,
      enableNotifications: json['enableNotifications'] as bool? ?? true,
      showBadge: json['showBadge'] as bool? ?? true,
    );
  }

  /// Create a copy with optional field overrides
  SettingsModel copyWith({
    bool? enableMotionAlerts,
    int? alertSensitivity,
    String? alertSound,
    bool? darkMode,
    int? refreshRateSeconds,
    bool? enableNotifications,
    bool? showBadge,
  }) {
    return SettingsModel(
      enableMotionAlerts: enableMotionAlerts ?? this.enableMotionAlerts,
      alertSensitivity: alertSensitivity ?? this.alertSensitivity,
      alertSound: alertSound ?? this.alertSound,
      darkMode: darkMode ?? this.darkMode,
      refreshRateSeconds: refreshRateSeconds ?? this.refreshRateSeconds,
      enableNotifications: enableNotifications ?? this.enableNotifications,
      showBadge: showBadge ?? this.showBadge,
    );
  }

  /// Get default settings
  static SettingsModel defaults() {
    return SettingsModel();
  }
}
